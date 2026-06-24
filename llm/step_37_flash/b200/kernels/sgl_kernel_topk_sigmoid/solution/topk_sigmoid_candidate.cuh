// Candidate native-CUDA implementation of sgl_kernel.topk_sigmoid for the captured
// Step-3.7-Flash-FP8 routing contract (fp32 gating [N,288], topk=8, num_experts=288,
// renormalize=True, fp32 correction_bias [288]).
//
// Optimization lever: upstream takes the two-launch workspace path for non-power-of-two
// expert counts (288). This candidate fuses sigmoid + biased-selection + top-8 + unbiased
// weights + renormalization into ONE kernel launch with no workspace allocation.
//
// Bit-faithfulness to the baseline (see docs/baseline_source.md):
//   score[e] = sigmoid(gating[r,e]) + bias[e]      (fp32, expf, same as moeSigmoid)
//   select top-k by score, descending; tie-break = lower expert index wins (cub::ArgMax)
//   weight   = score - bias[e]   (the unbiased sigmoid, exactly as moeTopK subtracts bias back)
//   if renormalize: weight[i] /= sum_{j} weight[j]   (sum accumulated in selection order)
//
// One block processes one token row. The block does k sequential argmax reductions over the
// num_experts scores held in shared memory, masking each selected expert with -inf. For the
// realistic Step-3.7 bias range this reproduces the baseline's genuine top-k selection
// (the baseline's -1.f sentinel in moeTopK only diverges when fewer than topk experts have
// score > -1, which does not occur for realistic router biases; see docs/baseline_source.md).
#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>  // CUDART_INF_F

namespace topk_sigmoid_candidate {

// Compile-time contract for the fused fast path.
constexpr int kNumExperts = 288;
constexpr int kTopK = 8;
constexpr int kBlockThreads = 128;  // 4 warps; enough parallelism for 288 experts, small for N=1

// Combine two (value, index) candidates with the baseline's tie-break:
// larger value wins; on equal value the smaller index wins (matches cub::ArgMax).
__device__ __forceinline__ void argmax_combine(float& best_v, int& best_i, float v, int i) {
  if (v > best_v || (v == best_v && i < best_i)) {
    best_v = v;
    best_i = i;
  }
}

// Fused topk_sigmoid kernel. grid.x == num_tokens, blockDim.x == kBlockThreads.
// gating: [N, num_experts] fp32 (row-major, contiguous)
// bias:   [num_experts] fp32 (always present for the captured contract)
// out_weights: [N, topk] fp32 (in-place output)
// out_indices: [N, topk] int32 (in-place output)
template <int NUM_EXPERTS, int TOPK, int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS) topk_sigmoid_fused_kernel(
    const float* __restrict__ gating,
    const float* __restrict__ bias,
    float* __restrict__ out_weights,
    int* __restrict__ out_indices,
    int num_tokens,
    bool renormalize) {
  const int row = blockIdx.x;
  if (row >= num_tokens) return;

  __shared__ float s_score[NUM_EXPERTS];   // biased selection score per expert
  __shared__ float s_red_val[BLOCK_THREADS];
  __shared__ int s_red_idx[BLOCK_THREADS];
  __shared__ float s_sel_w[TOPK];           // unbiased selected weights (selection order)
  __shared__ int s_sel_i[TOPK];

  const float* grow = gating + static_cast<long long>(row) * NUM_EXPERTS;
  const int tid = threadIdx.x;

  // Pass 1: sigmoid + bias -> biased selection score in shared memory.
  for (int e = tid; e < NUM_EXPERTS; e += BLOCK_THREADS) {
    float v = grow[e];
    v = 1.0f / (1.0f + expf(-v));  // sigmoid (matches moeSigmoid)
    v = v + bias[e];               // biased selection score
    s_score[e] = v;
  }
  __syncthreads();

  // Pass 2: k sequential argmax reductions with lower-index tie-break, masking selected.
  for (int k = 0; k < TOPK; ++k) {
    float best_v = -CUDART_INF_F;
    int best_i = NUM_EXPERTS;  // sentinel index (never beats a real expert on tie)
    for (int e = tid; e < NUM_EXPERTS; e += BLOCK_THREADS) {
      argmax_combine(best_v, best_i, s_score[e], e);
    }
    s_red_val[tid] = best_v;
    s_red_idx[tid] = best_i;
    __syncthreads();

    // Tree reduction over the block (BLOCK_THREADS must be a power of two).
    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        argmax_combine(s_red_val[tid], s_red_idx[tid], s_red_val[tid + stride], s_red_idx[tid + stride]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      const int e = s_red_idx[0];
      const float score = s_red_val[0];
      s_sel_i[k] = e;
      s_sel_w[k] = score - bias[e];  // unbiased weight (bias subtracted back, as in moeTopK)
      s_score[e] = -CUDART_INF_F;    // mask so it is not reselected
    }
    __syncthreads();
  }

  // Pass 3: renormalize (sum in selection order) and write outputs.
  if (tid == 0) {
    float* wrow = out_weights + static_cast<long long>(row) * TOPK;
    int* irow = out_indices + static_cast<long long>(row) * TOPK;
    float inv = 1.0f;
    if (renormalize) {
      float sum = 0.0f;
      for (int k = 0; k < TOPK; ++k) sum += s_sel_w[k];
      inv = 1.0f / sum;
    }
    for (int k = 0; k < TOPK; ++k) {
      wrow[k] = renormalize ? (s_sel_w[k] * inv) : s_sel_w[k];
      irow[k] = s_sel_i[k];
    }
  }
}

// Host launcher for the fused fast path. Assumes the validated contract
// (num_experts == kNumExperts, topk == kTopK, fp32 contiguous, bias != nullptr).
inline void launch_topk_sigmoid_candidate(
    const float* gating,
    const float* bias,
    float* out_weights,
    int* out_indices,
    int num_tokens,
    bool renormalize,
    cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(kBlockThreads);
  topk_sigmoid_fused_kernel<kNumExperts, kTopK, kBlockThreads>
      <<<grid, block, 0, stream>>>(gating, bias, out_weights, out_indices, num_tokens, renormalize);
}

// Host-side route decision (no tensor reads, no sync): returns true iff the fused fast path
// covers this call exactly; otherwise the caller must fall back to the recovered baseline.
inline bool candidate_covers(
    int num_experts,
    int topk,
    bool gating_is_fp32,
    bool gating_contiguous,
    bool weights_contiguous,
    bool indices_contiguous,
    bool has_bias) {
  return num_experts == kNumExperts && topk == kTopK && gating_is_fp32 && gating_contiguous &&
         weights_contiguous && indices_contiguous && has_bias;
}

}  // namespace topk_sigmoid_candidate
