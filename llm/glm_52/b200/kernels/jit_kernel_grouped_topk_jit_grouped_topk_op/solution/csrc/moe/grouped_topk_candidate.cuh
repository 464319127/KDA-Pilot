/*
 * Native-CUDA candidate for the fused biased grouped top-k MoE router.
 *
 * Same op contract and ABI as the recovered baseline
 * (baseline/csrc/moe/grouped_topk.cuh): single-group (num_expert_group=1,
 * topk_group=1) sigmoid+bias scoring, top-k by biased score with smaller-index
 * tie-break, un-biased sigmoid weights, renormalize, scaling factor;
 * destination-passing into topk_values[N,topk] f32 and topk_indices[N,topk] i32.
 *
 * Two kernels behind one shape-based dispatch:
 *   - LARGE-N path: one WARP PER TOKEN, several warps per block, register-
 *     resident (no shared memory). Every thread is productive and the block
 *     count drops; this wins once there is enough work to fill the SMs.
 *   - SMALL-N path: the recovered baseline kernel (one block per token, all
 *     threads compute the sigmoid pass in parallel). For tiny token counts the
 *     work is launch-floor / SFU-latency bound and per-token full-block
 *     parallelism beats warp-per-token, so the candidate dispatches to the
 *     baseline algorithm there (exact same result; no regression).
 *
 * The SMALL-N / off-domain fallback path IS the copied baseline kernel, so its
 * output is bit-identical to the baseline by construction. The LARGE-N warp path
 * reuses the baseline's exact selection math (fast_sigmoid, packed value+index,
 * renormalization reduction), so it matches the baseline on exact ordered top-k
 * indices and on weights within fp32 tolerance (validated by bench/correctness.py;
 * arithmetically it is the same sequence of ops). Unsupported parameters are
 * rejected exactly as the baseline.
 */
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, fp32_t

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace {

static constexpr int WARP_SIZE = 32;
static constexpr int MAX_TOPK = 8;

// ── shared selection helpers (bit-identical to the baseline) ─────────────────
__device__ __forceinline__ uint64_t pack_val_idx(float val, int32_t idx) {
  uint32_t val_bits = __float_as_uint(val);
  val_bits ^= (val_bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  uint32_t idx_bits = static_cast<uint32_t>(65535 - idx);
  return (static_cast<uint64_t>(val_bits) << 32) | idx_bits;
}

__device__ __forceinline__ void unpack_val_idx(uint64_t packed, float& val, int32_t& idx) {
  uint32_t idx_bits = static_cast<uint32_t>(packed & 0xFFFFFFFF);
  idx = static_cast<int32_t>(65535 - idx_bits);
  uint32_t val_bits = static_cast<uint32_t>(packed >> 32);
  val_bits ^= (val_bits & 0x80000000u) ? 0x80000000u : 0xffffffffu;
  val = __uint_as_float(val_bits);
}

__device__ __forceinline__ uint64_t warp_max_u64(uint64_t val) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    uint64_t other = __shfl_xor_sync(0xffffffffu, val, mask);
    val = max(val, other);
  }
  return val;
}

__device__ __forceinline__ float warp_sum_f32(float val) {
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, mask);
  }
  return val;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// ─────────────────────────────────────────────────────────────────────────────
// LARGE-N path: one warp per token, WarpsPerBlock tokens per block. Each lane
// owns experts {lane, lane+32, ...} (up to MaxExpertsPerLane) in registers.
// ─────────────────────────────────────────────────────────────────────────────
template <int WarpsPerBlock, int MaxExpertsPerLane>
__global__ void grouped_topk_warp_per_token_kernel(
    const float* __restrict__ scores,
    float* __restrict__ topk_values,
    int32_t* __restrict__ topk_indices,
    const float* __restrict__ bias,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t topk,
    bool renormalize,
    float scaling_factor) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp_in_block = threadIdx.x >> 5;
  const int64_t token_id = static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp_in_block;
  if (token_id >= num_tokens) return;  // whole warp returns together

  const float* __restrict__ token_scores = scores + token_id * num_experts;

  float sig[MaxExpertsPerLane];
  float biased[MaxExpertsPerLane];
#pragma unroll
  for (int j = 0; j < MaxExpertsPerLane; ++j) {
    const int e = lane + j * WARP_SIZE;
    if (e < num_experts) {
      const float s = fast_sigmoid(token_scores[e]);
      sig[j] = s;
      biased[j] = s + bias[e];
    } else {
      sig[j] = 0.0f;
      biased[j] = -FLT_MAX;
    }
  }

  int32_t winner_idx = 0;
  float winner_sig = 0.0f;
  for (int k = 0; k < topk; ++k) {
    float local_val = -FLT_MAX;
    int32_t local_idx = lane;
#pragma unroll
    for (int j = 0; j < MaxExpertsPerLane; ++j) {
      if (biased[j] > local_val) {
        local_val = biased[j];
        local_idx = lane + j * WARP_SIZE;
      }
    }
    const uint64_t best = warp_max_u64(pack_val_idx(local_val, local_idx));
    float best_val;
    int32_t best_idx;
    unpack_val_idx(best, best_val, best_idx);

    const int owner = best_idx & (WARP_SIZE - 1);
    float owned_sig = 0.0f;
#pragma unroll
    for (int j = 0; j < MaxExpertsPerLane; ++j) {
      if ((lane + j * WARP_SIZE) == best_idx) owned_sig = sig[j];
    }
    const float sel_sig = __shfl_sync(0xffffffffu, owned_sig, owner);

    if (lane == k) {
      winner_idx = best_idx;
      winner_sig = sel_sig;
    }
#pragma unroll
    for (int j = 0; j < MaxExpertsPerLane; ++j) {
      if (lane == owner && (lane + j * WARP_SIZE) == best_idx) biased[j] = -FLT_MAX;
    }
    __syncwarp();
  }

  const float weight = (lane < topk) ? winner_sig : 0.0f;
  const float divisor = renormalize ? (warp_sum_f32(weight) + 1e-20f) : 1.0f;
  if (lane < topk) {
    topk_values[token_id * topk + lane] = weight * scaling_factor / divisor;
    topk_indices[token_id * topk + lane] = winner_idx;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SMALL-N path: the recovered baseline kernel (one block per token, MaxExperts
// threads). Copied verbatim from baseline/csrc/moe/grouped_topk.cuh so the tiny-
// token regime keeps the baseline's full-block parallel sigmoid (it is the best
// available there — launch-floor / SFU bound). Identical numerics.
// ─────────────────────────────────────────────────────────────────────────────
template <int MaxExperts>
__global__ void grouped_topk_block_per_token_kernel(
    const float* __restrict__ scores,
    float* __restrict__ topk_values,
    int32_t* __restrict__ topk_indices,
    const float* __restrict__ bias,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t topk,
    bool renormalize,
    float scaling_factor) {
  __shared__ float smem_sigmoid[MaxExperts];
  __shared__ float smem_biased[MaxExperts];

  int64_t token_id = blockIdx.x;
  if (token_id >= num_tokens) return;

  int tid = threadIdx.x;
  const float* token_scores = scores + token_id * num_experts;

  float score_sig = -FLT_MAX;
  float score_biased = -FLT_MAX;
  if (tid < num_experts) {
    float raw = token_scores[tid];
    score_sig = fast_sigmoid(raw);
    score_biased = score_sig + bias[tid];
  }
  smem_sigmoid[tid] = score_sig;
  smem_biased[tid] = score_biased;
  __syncthreads();

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if (warp_id != 0) return;

  float* out_vals = topk_values + token_id * topk;
  int32_t* out_ids = topk_indices + token_id * topk;

  float selected_weights[MAX_TOPK];
  int32_t selected_ids[MAX_TOPK];

  for (int k = 0; k < topk; k++) {
    float my_max_val = -FLT_MAX;
    int32_t my_max_idx = 0;
    for (int i = lane_id; i < num_experts; i += WARP_SIZE) {
      float v = smem_biased[i];
      if (v > my_max_val) {
        my_max_val = v;
        my_max_idx = i;
      }
    }
    uint64_t packed = pack_val_idx(my_max_val, my_max_idx);
    uint64_t best = warp_max_u64(packed);
    float best_val;
    int32_t best_idx;
    unpack_val_idx(best, best_val, best_idx);

    selected_ids[k] = best_idx;
    selected_weights[k] = smem_sigmoid[best_idx];

    if (best_idx >= WARP_SIZE) {
      if (lane_id == 0) smem_biased[best_idx] = -FLT_MAX;
    } else {
      if (lane_id == best_idx) smem_biased[best_idx] = -FLT_MAX;
    }
    __syncwarp();
  }

  float weight = (lane_id < topk) ? selected_weights[lane_id] : 0.0f;
  float divisor = renormalize ? warp_sum_f32(weight) + 1e-20f : 1.0f;
  if (lane_id < topk) {
    out_ids[lane_id] = selected_ids[lane_id];
    out_vals[lane_id] = weight * scaling_factor / divisor;
  }
}

// ── dispatch tuning ──────────────────────────────────────────────────────────
// Optional warps-per-block override (read once). When set (1/2/4/8) it selects that
// warps-per-block for the warp path and lowers the token threshold — but ONLY within
// the production domain (the dispatch predicate below still hard-gates on
// production_domain, so the override can never route an off-domain input to the warp
// kernel). 0 = shape-based dispatch.
inline int warps_per_block_override() {
  static const int v = [] {
    const char* e = std::getenv("K09_WPB");
    if (!e) return 0;
    int x = std::atoi(e);
    return (x == 1 || x == 2 || x == 4 || x == 8) ? x : 0;
  }();
  return v;
}

// Below this token count the warp-per-token path cannot beat the baseline
// (launch-floor / SFU bound), so dispatch to the baseline kernel. Tuned from the
// per-N B200 sweep (see docs/dispatch.md).
static constexpr int64_t WARP_PATH_MIN_TOKENS = 768;

// ─────────────────────────────────────────────────────────────────────────────
// Launcher: identical ABI / verification / RuntimeChecks as the baseline.
// Covers the whole supported domain; unsupported parameters reject like baseline.
// ─────────────────────────────────────────────────────────────────────────────
void grouped_topk(
    tvm::ffi::TensorView scores,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView topk_values,
    tvm::ffi::TensorView topk_indices,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    bool renormalize,
    double scaling_factor) {
  using namespace host;

  SymbolicSize N{"num_tokens"};
  SymbolicSize E{"num_experts"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N, E}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(scores);
  TensorMatcher({E}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(bias);

  SymbolicSize K{"topk"};
  TensorMatcher({N, K}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(topk_values);
  TensorMatcher({N, K}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk_indices);

  int64_t num_tokens = N.unwrap();
  int64_t num_experts = E.unwrap();
  DLDevice device = device_.unwrap();

  RuntimeCheck(num_expert_group == 1 && topk_group == 1, "This kernel only supports num_expert_group=1, topk_group=1");
  RuntimeCheck(topk <= MAX_TOPK, "topk must be <= ", MAX_TOPK);
  RuntimeCheck(num_experts <= 512, "num_experts must be <= 512");

  if (num_tokens == 0) return;

  const float scale_f = static_cast<float>(scaling_factor);
  auto* score_ptr = static_cast<const float*>(scores.data_ptr());
  auto* bias_ptr = static_cast<const float*>(bias.data_ptr());
  auto* val_ptr = static_cast<float*>(topk_values.data_ptr());
  auto* idx_ptr = static_cast<int32_t*>(topk_indices.data_ptr());

  const int ov = warps_per_block_override();
  // The warp-per-token fast path runs ONLY on the captured production domain.
  // Contiguity and fp32 are already enforced by the TensorMatcher verifies above
  // (non-contiguous / non-fp32 are rejected identically to the baseline before
  // reaching here), so they are not re-tested in this predicate. Every other
  // baseline-supported case — num_experts != 256, topk != 8, renormalize == false,
  // scaling_factor != 1, or num_tokens < WARP_PATH_MIN_TOKENS — falls back to the
  // recovered baseline kernel below.
  const bool production_domain =
      (num_experts == 256) && (topk == 8) && (num_expert_group == 1) &&
      (topk_group == 1) && renormalize && (scale_f == 1.0f);
  // production_domain is a HARD gate: the warp path is only ever taken inside the
  // captured production domain. The K09_WPB tuning override can only vary
  // warps-per-block / lower the token threshold WITHIN that domain — it can never
  // route an off-domain input (E!=256, topk!=8, renormalize=False, scaling!=1, etc.)
  // to the warp kernel; those always fall back to the baseline kernel below.
  const bool use_warp_path = production_domain && ((ov != 0) || (num_tokens >= WARP_PATH_MIN_TOKENS));

  if (!use_warp_path) {
    // Baseline block-per-token path for the small-N / launch-floor regime.
    const uint32_t grid = static_cast<uint32_t>(num_tokens);
    if (num_experts <= 128) {
      LaunchKernel(grid, 128u, device)(grouped_topk_block_per_token_kernel<128>,
          score_ptr, val_ptr, idx_ptr, bias_ptr, num_tokens, num_experts, topk, renormalize, scale_f);
    } else if (num_experts <= 256) {
      LaunchKernel(grid, 256u, device)(grouped_topk_block_per_token_kernel<256>,
          score_ptr, val_ptr, idx_ptr, bias_ptr, num_tokens, num_experts, topk, renormalize, scale_f);
    } else {
      LaunchKernel(grid, 512u, device)(grouped_topk_block_per_token_kernel<512>,
          score_ptr, val_ptr, idx_ptr, bias_ptr, num_tokens, num_experts, topk, renormalize, scale_f);
    }
    return;
  }

  // Large-N warp-per-token path. W tuned from the per-N sweep (fewer blocks help
  // the boundary region; the large region is dominated by per-token throughput).
  int wpb;
  if (ov != 0) {
    wpb = ov;
  } else {
    wpb = (num_tokens <= 1280) ? 8 : 4;
  }
  const int epl_tier = (num_experts <= 128) ? 4 : (num_experts <= 256 ? 8 : 16);

  auto run = [&](auto Wc, auto Ec) {
    constexpr int W = decltype(Wc)::value;
    constexpr int EPL = decltype(Ec)::value;
    const uint32_t grid = static_cast<uint32_t>((num_tokens + W - 1) / W);
    const uint32_t block = static_cast<uint32_t>(W * WARP_SIZE);
    LaunchKernel(grid, block, device)(
        grouped_topk_warp_per_token_kernel<W, EPL>,
        score_ptr, val_ptr, idx_ptr, bias_ptr,
        num_tokens, num_experts, topk, renormalize, scale_f);
  };
  using std::integral_constant;
#define K09_DISPATCH_EPL(W)                                                                       \
  do {                                                                                            \
    if (epl_tier == 4) { run(integral_constant<int, W>{}, integral_constant<int, 4>{}); return; } \
    if (epl_tier == 8) { run(integral_constant<int, W>{}, integral_constant<int, 8>{}); return; } \
    run(integral_constant<int, W>{}, integral_constant<int, 16>{}); return;                       \
  } while (0)
  switch (wpb) {
    case 1: K09_DISPATCH_EPL(1);
    case 2: K09_DISPATCH_EPL(2);
    case 4: K09_DISPATCH_EPL(4);
    default: K09_DISPATCH_EPL(8);
  }
#undef K09_DISPATCH_EPL
}

}  // namespace
