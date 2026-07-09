// Native-CUDA decode (M == 1) FP8 block-scaled GEMV for the
// deep_gemm_fp8_fp8_bf16_nt interface on B200 (SM100).
//
// Semantics (NT, identical to the recovered baseline):
//   out[0, n] = sum_k float(A[0, k]) * scale_a(k/128) * float(B[n, k]) * scale_b(n, k/128)
// A is [1, K] fp8_e4m3 row-major contiguous, B is [N, K] fp8_e4m3 row-major contiguous,
// out is [1, N] bf16 contiguous, written in place, launched on the current CUDA stream.
//
// Scales are packed UE8M0 int32 (four 8-bit exponents per word, little-endian byte
// order): for K-group kg = k/128, word index = kg >> 2, byte lane = kg & 3, and
// scale = 2^(E-127) = __int_as_float(E << 23). Packed layout is MN-major: element
// (row, word) lives at row*stride0 + word*stride1 (int32 units); the M == 1 A-scale
// row uses word stride stride1, B-scales use per-row stride0 with stride(0) == 1.
//
// The regime is memory-bound (~2 FLOP per B byte): stream B exactly once. One warp
// owns one output column; each lane consumes 16 fp8 (one uint4) per iteration, so a
// warp covers 512 B = four 128-wide K-groups per step. A and its dequantized per-group
// scales are staged once per CTA in shared memory. All four B-scale exponents a warp
// needs per step share ONE int32 word (kg>>2 == step), so lane 0 loads it and the warp
// takes it from a shuffle broadcast instead of 32 duplicate loads.
//
// Two launch configurations behind one kernel template:
//   SplitK = 1: four warps -> four columns per CTA (wide-N rows).
//   SplitK = 2: four warps -> two columns per CTA, each column split into two K-halves
//               reduced through shared memory. Used when N alone cannot fill the GPU
//               (e.g. N=512 -> 128 CTAs on 148 SMs with SplitK=1, 256 CTAs with 2).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "fp8_gemv_common.cuh"

namespace {

constexpr int kWarpsPerCta = 4;
constexpr int kThreads = kWarpsPerCta * 32;


// SplitK == 1: warp w handles column blockIdx.x*4 + w over the full K range.
// SplitK == 2: warps (2c, 2c+1) handle column blockIdx.x*2 + c; warp parity picks the
//              K half. Requires K % 1024 == 0 so each half stays a 512-byte multiple.
template <int SplitK>
__global__ void __launch_bounds__(kThreads) decode_m1_gemv_kernel(
    const uint8_t* __restrict__ A,     // [K]
    const int32_t* __restrict__ Asf,   // packed UE8M0, row 0 of [1, W]
    const uint8_t* __restrict__ B,     // [N, K]
    const int32_t* __restrict__ Bsf,   // packed UE8M0, [N, W]
    __nv_bfloat16* __restrict__ out,   // [N]
    int N, int K,
    int Asf_s1, int Bsf_s0, int Bsf_s1) {
    extern __shared__ unsigned char smem[];
    uint8_t* sA = smem;                                       // K bytes of A
    float* sAscale = reinterpret_cast<float*>(sA + K);        // K/128 dequantized A scales
    float* sPartial = sAscale + (K >> 7);                     // SplitK partials (4 floats)

    const int tid = threadIdx.x;
    const int num_groups = K >> 7;

    for (int i = tid * 16; i < K; i += kThreads * 16) {
        *reinterpret_cast<uint4*>(sA + i) = *reinterpret_cast<const uint4*>(A + i);
    }
    for (int kg = tid; kg < num_groups; kg += kThreads) {
        uint32_t w = static_cast<uint32_t>(Asf[(kg >> 2) * Asf_s1]);
        sAscale[kg] = ue8m0_byte_to_scale(w, kg & 3);
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;

    int n, step_begin, step_end;
    if (SplitK == 1) {
        n = blockIdx.x * kWarpsPerCta + warp;
        step_begin = 0;
        step_end = (num_groups + 3) >> 2;                     // ceil: tail lanes masked
    } else {
        n = blockIdx.x * (kWarpsPerCta / 2) + (warp >> 1);
        const int steps_total = num_groups >> 2;              // K % 1024 == 0 -> exact
        const int half = steps_total >> 1;
        step_begin = (warp & 1) * half;
        step_end = (warp & 1) ? steps_total : half;
    }
    // No early return: the SplitK==2 epilogue has a CTA barrier every thread must
    // reach, so out-of-range columns (possible only in the last CTA) just skip work.
    const bool valid = n < N;
    const uint8_t* __restrict__ Brow = B + static_cast<size_t>(valid ? n : 0) * K;
    const int32_t* __restrict__ BsfRow = Bsf + static_cast<size_t>(valid ? n : 0) * Bsf_s0;

    // The regime is memory-latency-bound at warp-per-column occupancy (a single
    // outstanding 512 B coalesced load per warp cannot hide the load latency), so the
    // main loop must be a pure load+math body the compiler can software-pipeline.
    // All packed B-scale words a warp can ever need (word index == step, at most 12
    // for K=6144) are read up front — lane w holds word w — and each step takes its
    // word from a register shuffle instead of a fresh load, keeping the per-step
    // dependency chains free of memory operations besides the B stream itself.
    float acc = 0.0f;
    if (valid) {
        const int num_words = (num_groups + 3) >> 2;
        const uint32_t my_word = (lane < num_words)
            ? static_cast<uint32_t>(BsfRow[lane * Bsf_s1]) : 0u;
#pragma unroll 4
        for (int step = step_begin; step < step_end; ++step) {
            const uint32_t b_word = __shfl_sync(0xFFFFFFFFu, my_word, step);
            const int off = step * 512 + lane * 16;
            if (off >= K) continue;                           // K % 512 != 0 tail lanes

            const uint4 bw = *reinterpret_cast<const uint4*>(Brow + off);
            const uint4 aw = *reinterpret_cast<const uint4*>(sA + off);
            const float part = dot16_fp8(aw, bw);

            const int kg = step * 4 + (lane >> 3);
            acc += part * sAscale[kg] * ue8m0_byte_to_scale(b_word, lane >> 3);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }

    if (SplitK == 1) {
        if (valid && lane == 0) out[n] = __float2bfloat16(acc);
    } else {
        if (lane == 0) sPartial[warp] = acc;
        __syncthreads();
        if (valid && lane == 0 && (warp & 1) == 0) {
            out[n] = __float2bfloat16(sPartial[warp] + sPartial[warp + 1]);
        }
    }
}

}  // namespace

// Dispatch policy between the two launch configurations lives here (the host knows
// N and K); the Python-side predicate only gates layout support. Launched on the
// current CUDA stream; out is written in place (destination passing).
void decode_m1_gemv(torch::Tensor A, torch::Tensor Asf,
                    torch::Tensor B, torch::Tensor Bsf,
                    torch::Tensor out) {
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));
    TORCH_CHECK(A.size(0) == 1 && (K & 127) == 0, "decode_m1_gemv: M must be 1, K a multiple of 128");
    TORCH_CHECK(((K >> 7) + 3) / 4 <= 32,
                "decode_m1_gemv: at most 32 packed scale words supported (K <= 16384); the\n"
                "warp-shuffle scale preload selects words by source lane");
    TORCH_CHECK(B.size(1) == K && out.size(0) == 1 && out.size(1) == N, "decode_m1_gemv: shape mismatch");
    TORCH_CHECK(Asf.get_device() == A.get_device() && B.get_device() == A.get_device()
                    && Bsf.get_device() == A.get_device() && out.get_device() == A.get_device(),
                "decode_m1_gemv: all tensors must be on one CUDA device");

    // In a multi-GPU process the current device may differ from the tensors'
    // device; the stream below must belong to the tensors' device.
    const c10::cuda::CUDAGuard device_guard(A.device());
    const size_t smem = static_cast<size_t>(K) + static_cast<size_t>(K >> 7) * sizeof(float)
                        + kWarpsPerCta * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    // SplitK=2 doubles the resident warp count for rows whose column count alone
    // cannot occupy the GPU (measured crossover: helps up to N=3072 -> 768
    // warp-per-column CTAs on 148 SMs; hurts wide-N rows where occupancy is already
    // saturated and the split only shortens per-warp streams and adds a barrier).
    // Each K half must stay a 512-byte multiple.
    const bool use_split_k = ((K & 1023) == 0) && (K >= 2048) && (N <= 3072);
    const auto* a_ptr = reinterpret_cast<const uint8_t*>(A.data_ptr());
    const auto* b_ptr = reinterpret_cast<const uint8_t*>(B.data_ptr());
    auto* o_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const int asf_s1 = static_cast<int>(Asf.stride(1));
    const int bsf_s0 = static_cast<int>(Bsf.stride(0));
    const int bsf_s1 = static_cast<int>(Bsf.stride(1));

    if (use_split_k) {
        const int blocks = (N + (kWarpsPerCta / 2) - 1) / (kWarpsPerCta / 2);
        decode_m1_gemv_kernel<2><<<blocks, kThreads, smem, stream>>>(
            a_ptr, Asf.data_ptr<int32_t>(), b_ptr, Bsf.data_ptr<int32_t>(), o_ptr,
            N, K, asf_s1, bsf_s0, bsf_s1);
    } else {
        const int blocks = (N + kWarpsPerCta - 1) / kWarpsPerCta;
        decode_m1_gemv_kernel<1><<<blocks, kThreads, smem, stream>>>(
            a_ptr, Asf.data_ptr<int32_t>(), b_ptr, Bsf.data_ptr<int32_t>(), o_ptr,
            N, K, asf_s1, bsf_s0, bsf_s1);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_m1_gemv", &decode_m1_gemv, "M=1 FP8 block-scaled GEMV (B200/SM100)");
}
