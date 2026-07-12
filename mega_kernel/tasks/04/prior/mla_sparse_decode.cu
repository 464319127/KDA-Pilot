// Native-CUDA sparse-MLA absorbed decode candidate for small batch (B=1 first).
//
// Ported with attribution from the completed sibling run of this task slug
// (worktree glm_52__sglang_unified_attention_with_output-20260707-203446-78920,
// solution/mla_sparse_decode.cu); re-validated and re-measured in this worktree.
//
// NCU showed the baseline fmhaSm100f decode launches only 32 CTAs on 148 SMs at B=1
// (0.216 waves, 2.8% compute SOL): grid/tail-latency-bound + under-occupied. This
// candidate is a two-stage flash-decode ("split-KV") that launches many more CTAs and
// hides the scattered gather latency with warp-parallel work:
//   stage 1 (partials): grid (S, H, B), 4 warps/CTA. Each warp runs an online-softmax
//            over its stride-4 subset of one TILE-token chunk for one (b,h): the 576-dim
//            QK dot is a warp-shuffle reduction, the 512-dim output accumulator lives in
//            registers (16/lane), so there are NO per-token block syncs. The 4 warp
//            partials are combined once and written as (m,l,o[512]).
//   stage 2 (combine):  grid (V_DIM/DTILE, H, B) — combine the S chunk-partials per (b,h).
//
// Semantics (validated): block_tables are FLAT physical KV slot ids
// (slot -> kv_cache[slot/64,0,slot%64,:]); score = bmm1_scale * dot over the full
// absorbed HEAD_DIM=576; softmax over valid rows (j < min(seq_lens[b],top_k), slot in
// [0,num_slots)); out = softmax @ v with v = k[:512]; bf16 out; fp8 e4m3 value-domain
// dequant. Plain CUDA C++/SIMT (no CUTLASS/tensor cores — the lever is occupancy).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>

namespace {

constexpr int HEAD_DIM = 576;   // kv_lora_rank(512) + qk_rope_head_dim(64)
constexpr int V_DIM = 512;      // kv_lora_rank (V = latent slice)
constexpr int TILE = 32;        // top-k tokens per stage-1 chunk (more chunks -> more CTAs)
constexpr int NWARPS = 4;
constexpr int BLK = NWARPS * 32;
constexpr int QN = (HEAD_DIM + 31) / 32;  // 18 head dims per lane
constexpr int VN = (V_DIM + 31) / 32;     // 16 output dims per lane
constexpr int RED_WARPS = 8;    // warps per stage-2 CTA (one warp per output element)

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Stage 1: warp-parallel online-softmax partials over one TILE-token chunk for one (b,h).
__global__ void mla_partials(
        const __nv_fp8_e4m3* __restrict__ query,   // [B,1,H,576]
        const __nv_fp8_e4m3* __restrict__ kv,      // flat [num_slots,576]
        const int* __restrict__ block_tables,      // [B,1,top_k]
        const int* __restrict__ seq_lens,          // [B]
        float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_o,
        int H, int S, int tile, int top_k, int num_slots, float bmm1_scale) {
    const int chunk = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;

    // load this lane's q dims (same q for all warps; redundant load is cheap)
    const __nv_fp8_e4m3* qp = query + ((long)(b * H + h)) * HEAD_DIM;
    float q_reg[QN];
#pragma unroll
    for (int i = 0; i < QN; ++i) {
        int d = lane + 32 * i;
        q_reg[i] = (d < HEAD_DIM) ? float(qp[d]) : 0.0f;
    }

    const int L = min(seq_lens[b], top_k);
    const int start = chunk * tile, end = min(start + tile, L);
    const int* bt = block_tables + (long)b * top_k;

    float m = -CUDART_INF_F, l = 0.0f;
    float o_reg[VN];
#pragma unroll
    for (int i = 0; i < VN; ++i) o_reg[i] = 0.0f;

    // warp w handles tokens start + w, start + w + NWARPS, ...
    for (int j = start + wid; j < end; j += NWARPS) {
        int slot = bt[j];
        if (slot < 0 || slot >= num_slots) continue;
        const __nv_fp8_e4m3* kp = kv + (long)slot * HEAD_DIM;
        float k_reg[QN];
#pragma unroll
        for (int i = 0; i < QN; ++i) {
            int d = lane + 32 * i;
            k_reg[i] = (d < HEAD_DIM) ? float(kp[d]) : 0.0f;
        }
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < QN; ++i) dot += q_reg[i] * k_reg[i];
        float score = warp_sum(dot) * bmm1_scale;
        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * corr + p;
#pragma unroll
        for (int i = 0; i < VN; ++i) o_reg[i] = o_reg[i] * corr + p * k_reg[i];  // k_reg[0..15] = v dims
        m = m_new;
    }

    // combine the NWARPS warp partials -> one chunk partial
    __shared__ float sm_m[NWARPS], sm_l[NWARPS], sm_o[NWARPS][V_DIM];
    if (lane == 0) { sm_m[wid] = m; sm_l[wid] = l; }
#pragma unroll
    for (int i = 0; i < VN; ++i) {
        int d = lane + 32 * i;
        if (d < V_DIM) sm_o[wid][d] = o_reg[i];
    }
    __syncthreads();

    // An EMPTY chunk (no valid token: start>=L, or all slots invalid) leaves every warp
    // max at -inf. Guard the combine so it never evaluates exp(-inf - -inf)=NaN — write the
    // neutral partial (m=-inf, l=0, o=0) instead, which stage 2 skips.
    const long bh = (long)(b * H + h);
    long idx = bh * S + chunk;                        // part_m / part_l: [bh, chunk]
    if (threadIdx.x == 0) {
        float M = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) M = fmaxf(M, sm_m[w]);
        float denom = 0.0f;
        if (M > -CUDART_INF_F)
#pragma unroll
            for (int w = 0; w < NWARPS; ++w) denom += __expf(sm_m[w] - M) * sm_l[w];
        part_m[idx] = M; part_l[idx] = denom;         // empty chunk -> (-inf, 0)
    }
    // part_o layout is [bh, dim, chunk] so stage-2's per-output warp reads consecutive
    // chunks (coalesced). Each thread combines its output dims across the warps.
    for (int d = threadIdx.x; d < V_DIM; d += BLK) {
        float M = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) M = fmaxf(M, sm_m[w]);
        float acc = 0.0f;
        if (M > -CUDART_INF_F)
#pragma unroll
            for (int w = 0; w < NWARPS; ++w) acc += __expf(sm_m[w] - M) * sm_o[w][d];
        part_o[(bh * V_DIM + d) * S + chunk] = acc;   // empty chunk -> 0
    }
}

// Stage 2: one WARP per output element (b,h,d) combines the S chunk-partials.
// B*H*V_DIM warps fill the GPU so the scattered partial reads are latency-hidden
// (the previous 1-warp-per-CTA version ran at 1.5% occupancy / 16us).
__global__ void mla_reduce(
        const float* __restrict__ part_m, const float* __restrict__ part_l,
        const float* __restrict__ part_o, __nv_bfloat16* __restrict__ out,
        int total, int S) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= total) return;
    const int d = warp % V_DIM;
    const long bh = warp / V_DIM;                 // = b*H + h
    const long base = bh * S;                     // part_m / part_l: [bh, chunk]
    const long obase = (bh * V_DIM + d) * S;      // part_o: [bh, dim, chunk] (coalesced)

    float m_local = -CUDART_INF_F;
    for (int c = lane; c < S; c += 32) m_local = fmaxf(m_local, part_m[base + c]);
    float M = m_local;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));

    float denom = 0.0f, acc = 0.0f;
    for (int c = lane; c < S; c += 32) {
        float mc = part_m[base + c];
        if (!(mc > -CUDART_INF_F)) continue;   // skip empty/non-finite chunk (mc==-inf or NaN)
        float w = __expf(mc - M);
        denom += w * part_l[base + c];
        acc += w * part_o[obase + c];
    }
    denom = warp_sum(denom);
    acc = warp_sum(acc);
    if (lane == 0)
        out[bh * V_DIM + d] =
            __float2bfloat16((M > -CUDART_INF_F && denom > 0.0f) ? acc / denom : 0.0f);
}

}  // namespace

// out: bf16 [B,1,H,512] contiguous. Launches on the current CUDA stream.
void mla_sparse_decode(torch::Tensor query, torch::Tensor kv_cache,
                       torch::Tensor block_tables, torch::Tensor seq_lens,
                       torch::Tensor workspace, torch::Tensor out,
                       double bmm1_scale, int64_t max_seq_len, int64_t tile_in) {
    const int B = query.size(0), H = query.size(2);
    const int num_slots = kv_cache.size(0) * kv_cache.size(2);
    const int top_k = block_tables.size(-1);
    const int Lmax = std::min((int)max_seq_len, top_k);
    const int tile = (tile_in > 0) ? (int)tile_in : TILE;
    const int S = (Lmax + tile - 1) / tile;

    float* ws = reinterpret_cast<float*>(workspace.data_ptr());
    float* part_m = ws;
    float* part_l = part_m + (long)B * H * S;
    float* part_o = part_l + (long)B * H * S;

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 g1(S, H, B);
    mla_partials<<<g1, BLK, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(query.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(kv_cache.data_ptr()),
        block_tables.data_ptr<int>(), seq_lens.data_ptr<int>(),
        part_m, part_l, part_o, H, S, tile, top_k, num_slots, (float)bmm1_scale);
    const int total = B * H * V_DIM;                 // one warp per output element
    const int rblk = RED_WARPS * 32;
    const int g2 = (total + RED_WARPS - 1) / RED_WARPS;
    mla_reduce<<<g2, rblk, 0, stream>>>(
        part_m, part_l, part_o, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), total, S);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_sparse_decode", &mla_sparse_decode, "native sparse-MLA decode (B=1 fast path)");
}
