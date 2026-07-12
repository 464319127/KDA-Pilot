// Fused DSA decode-wrapper prep for the native T=1 path (new in this run).
//
// The captured SGLang wrapper (`_forward_trtllm` fp8 branch) runs, per step:
//   mla_rope_quantize_fp8 (RoPE the rope bands + quantize bf16->fp8)
//   -> concat(q_nope_fp8, q_rope_fp8) into the [T,1,H,576] query
//   -> concat(k_nope_fp8, k_rope_fp8) + scatter into the paged KV pool (save_kv_cache)
//   -> trtllm_batch_decode_with_kv_cache_mla
// i.e. ~5 small launches + intermediates before the decode kernel. This kernel fuses
// all of the prep into ONE launch: strided bf16 gathers (the captured q_nope is a
// head-major slab view, q_rope/k_rope live at storage offsets of fused projection
// buffers), interleaved (is_neox=false) RoPE, fp8-E4M3 quantization, the fused query
// assembly, and the current-token KV-pool store. The attention core then runs as the
// native two-stage split-KV decode (solution/mla_sparse_decode.cu) on the pool.
//
// RoPE convention (validated against flashinfer.rope.mla_rope_quantize_fp8 by
// bench/sanity_oracle.py::case_rope_matches_flashinfer, is_neox=false interleaved):
//   cos_sin_cache row for position p is [cos(0..R/2), sin(0..R/2)] (R = rope dim);
//   pair (2i, 2i+1): x0' = x0*cos[i] - x1*sin[i];  x1' = x1*cos[i] + x0*sin[i].
// Rotation is computed in fp32 on the bf16 inputs, then quantized (RNE, saturating)
// to fp8 — matching the baseline op's compute path.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int KV_LORA = 512;
constexpr int ROPE = 64;
constexpr int DIM = KV_LORA + ROPE;  // 576
constexpr int BLK = 256;

__device__ __forceinline__ float rope_rotate(
        const __nv_bfloat16* band, int r, const float* cs, int half) {
    // band: one token/head's rope band (contiguous innermost, R values, fp32 math)
    const float c = cs[r >> 1];
    const float s = cs[half + (r >> 1)];
    const float x0 = __bfloat162float(band[r & ~1]);
    const float x1 = __bfloat162float(band[r | 1]);
    return (r & 1) ? (x1 * c + x0 * s) : (x0 * c - x1 * s);
}

// One CTA per token. Fuses: q assembly (nope copy + rope rotate) -> query_out fp8
// [N,1,H,576] contiguous; k assembly -> pool row at current_slots[t].
__global__ void fused_wrapper_prep(
        const __nv_bfloat16* __restrict__ q_nope,  // [N,H,512] strides (sqn0,sqn1,1)
        const __nv_bfloat16* __restrict__ q_rope,  // [N,H,64]  strides (sqr0,sqr1,1)
        const __nv_bfloat16* __restrict__ k_nope,  // [N,512]   strides (skn0,1)
        const __nv_bfloat16* __restrict__ k_rope,  // [N,64]    strides (skr0,1)
        const float* __restrict__ cos_sin,         // [num_pos,64] contiguous
        const int* __restrict__ pos_ids,           // [N]
        const long* __restrict__ current_slots,    // [N]
        __nv_fp8_e4m3* __restrict__ query_out,     // [N,1,H,576] contiguous
        __nv_fp8_e4m3* __restrict__ pool,          // flat [num_slots,576]
        int H,
        long sqn0, long sqn1, long sqr0, long sqr1, long skn0, long skr0) {
    const int t = blockIdx.x;
    const float* cs = cos_sin + (long)pos_ids[t] * ROPE;

    // q: H*576 output elements, laid out head-major in query_out.
    const int qtotal = H * DIM;
    for (int idx = threadIdx.x; idx < qtotal; idx += BLK) {
        const int h = idx / DIM, d = idx % DIM;
        float v;
        if (d < KV_LORA) {
            v = __bfloat162float(q_nope[(long)t * sqn0 + (long)h * sqn1 + d]);
        } else {
            v = rope_rotate(q_rope + (long)t * sqr0 + (long)h * sqr1, d - KV_LORA, cs, ROPE / 2);
        }
        query_out[(long)t * qtotal + idx] = __nv_fp8_e4m3(v);
    }

    // k: one 576-wide row scattered into the pool (save_kv_cache semantics).
    __nv_fp8_e4m3* dst = pool + (long)current_slots[t] * DIM;
    for (int d = threadIdx.x; d < DIM; d += BLK) {
        float v;
        if (d < KV_LORA) {
            v = __bfloat162float(k_nope[(long)t * skn0 + d]);
        } else {
            v = rope_rotate(k_rope + (long)t * skr0, d - KV_LORA, cs, ROPE / 2);
        }
        dst[d] = __nv_fp8_e4m3(v);
    }
}

}  // namespace

// Launches on the current CUDA stream. query_out fp8 [N,1,H,576] contiguous;
// pool is the fp8 KV pool viewed flat as [num_slots, 576].
void mla_wrapper_prep(torch::Tensor q_nope, torch::Tensor q_rope,
                      torch::Tensor k_nope, torch::Tensor k_rope,
                      torch::Tensor cos_sin_cache, torch::Tensor pos_ids,
                      torch::Tensor current_slots, torch::Tensor query_out,
                      torch::Tensor pool) {
    const int N = q_nope.size(0), H = q_nope.size(1);
    TORCH_CHECK(q_nope.size(2) == KV_LORA && q_rope.size(2) == ROPE,
                "wrapper prep expects kv_lora=512 rope=64");
    TORCH_CHECK(q_nope.stride(2) == 1 && q_rope.stride(2) == 1 &&
                k_nope.stride(1) == 1 && k_rope.stride(1) == 1,
                "innermost dims must be unit-stride (captured contract)");
    TORCH_CHECK(query_out.is_contiguous() && pool.stride(-1) == 1, "outputs must be dense");
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_wrapper_prep<<<N, BLK, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_rope.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_nope.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_rope.data_ptr()),
        cos_sin_cache.data_ptr<float>(), pos_ids.data_ptr<int>(),
        current_slots.data_ptr<long>(),
        reinterpret_cast<__nv_fp8_e4m3*>(query_out.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(pool.data_ptr()),
        H, q_nope.stride(0), q_nope.stride(1), q_rope.stride(0), q_rope.stride(1),
        k_nope.stride(0), k_rope.stride(0));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_wrapper_prep", &mla_wrapper_prep,
          "fused DSA wrapper prep: strided gather + interleaved RoPE + fp8 quant + KV store");
}
