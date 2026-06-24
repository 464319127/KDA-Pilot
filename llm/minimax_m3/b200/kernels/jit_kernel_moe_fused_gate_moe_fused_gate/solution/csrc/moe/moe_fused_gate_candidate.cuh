/*
 * Native-CUDA candidate for SGLang's fused MoE gating op `moe_fused_gate`.
 *
 * Same op contract and ABI as the recovered baseline
 * (baseline/csrc/moe/moe_fused_gate.cuh): per-expert score = scoring_func(input),
 * biased = score + bias; select topk_routed = topk - num_fused_shared_experts
 * experts by iterative arg-max on the BIASED score in descending order with a
 * smaller-index tie-break; the emitted weight uses the UN-biased score; the last
 * num_fused_shared_experts output slots are shared experts; renormalize by the
 * routed-score sum and apply the routed scaling factor. Destination-passing into
 * a caller-preallocated output[N,topk] f32 and indices[N,topk] i32.
 *
 * Why this file exists: the baseline small-token (decode) kernel is instantiated
 * with 8 warp slots but only 4 warps run for num_experts=128, so its cross-warp
 * reduction reads uninitialized shared memory (warp_maxs[4..7] / warp_experts[4..7]).
 * That feeds a garbage expert index into an out-of-bounds shared-memory write and
 * faults with an illegal memory access on a COLD CUDA context. This candidate
 * replaces the entire num_experts=128 production path with a cold-safe
 * warp-per-token kernel that uses NO shared memory at all: every reduction is an
 * intra-warp shuffle over the full 32-lane warp, so there is no uninitialized read
 * and it is correct as the very first launch on a fresh context.
 *
 * Dispatch:
 *   - In the captured production domain (num_experts=128, topk=5, scoring_func=0,
 *     num_fused_shared_experts=1, renormalize, routed_scaling_factor=2, apply on
 *     output) -> the cold-safe warp-per-token candidate kernel below.
 *   - Every other (off-domain) configuration -> the recovered baseline behaviour,
 *     BIT-IDENTICAL (baseline kernels copied verbatim into the `fallback` namespace).
 *     The fallback is NOT a general UB fix: it reproduces the baseline's decode bug for
 *     off-domain small-token configs with num_experts<=224 (e.g. E=128 with non-captured
 *     scalars such as topk!=5). The candidate removes the UB only for the captured
 *     production config, which is always handled by the cold-safe kernel above. See
 *     docs/dispatch.md for the scoped safety statement.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Shared scoring definition (used by both the candidate and the fallback).
// ─────────────────────────────────────────────────────────────────────────────
enum class ScoringFunc : uint32_t {
  kSigmoid = 0,
  kSqrtSoftplus = 1,
};

template <ScoringFunc kScoringFunc>
__device__ __forceinline__ float compute_score(float x) {
  if constexpr (kScoringFunc == ScoringFunc::kSigmoid) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    return 1.0f / (1.0f + expf(-x));
  } else {
    // sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
    float softplus = log1pf(expf(x));
    return sqrtf(softplus);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recovered baseline kernels, copied verbatim into a `fallback` namespace so the
// off-domain path is bit-identical to the baseline. Renamed only to avoid symbol
// clashes with the candidate kernel; the bodies are unchanged.
// ─────────────────────────────────────────────────────────────────────────────
namespace fallback {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpsPerCTA = 6;
constexpr uint32_t kSmallTokenThreshold = 512;
constexpr uint32_t kMaxExperts = 512;
constexpr uint32_t kMaxTopK = 16;

struct MoEFusedGateParams {
  const float* __restrict__ input;
  const float* __restrict__ bias;
  float* __restrict__ output;
  int32_t* __restrict__ indices;
  uint32_t num_rows;
  uint32_t num_experts;
  uint32_t topk;
  uint32_t num_fused_shared_experts;
  bool renormalize;
  float routed_scaling_factor;
  bool apply_routed_scaling_factor_on_output;
};

template <uint32_t kWarpsPerToken, ScoringFunc kScoringFunc>
__global__ void moe_fused_gate_kernel_small_token(const MoEFusedGateParams __grid_constant__ params) {
  const auto& [input, bias, output, indices, num_rows, num_experts, topk, num_fused_shared_experts, renormalize, routed_scaling_factor, apply_routed_scaling_factor_on_output] =
      params;

  uint32_t row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  // number of routed experts to select (excluding fused shared experts)
  const uint32_t topk_routed = topk - num_fused_shared_experts;

  uint32_t tid = threadIdx.x;
  uint32_t warp_id = tid / kWarpSize;
  uint32_t lane_id = tid % kWarpSize;

  extern __shared__ float shared_mem[];
  float* shared_scores = shared_mem;
  float* shared_original_scores = shared_mem + num_experts;

  // For warp-level reduction
  __shared__ float warp_maxs[kWarpsPerToken];
  __shared__ int warp_experts[kWarpsPerToken];
  __shared__ int selected_experts[kMaxTopK];

  for (uint32_t e = tid; e < num_experts; e += blockDim.x) {
    float input_val = input[row_idx * num_experts + e];
    float bias_val = bias[e];
    float score_val = compute_score<kScoringFunc>(input_val);
    float biased_val = score_val + bias_val;
    shared_scores[e] = biased_val;
    shared_original_scores[e] = score_val;
  }

  __syncthreads();

  // only select topk_routed experts (excluding shared experts)
  for (uint32_t k = 0; k < topk_routed; k++) {
    float my_val = -FLT_MAX;
    int my_expert = -1;
    for (uint32_t e = tid; e < num_experts; e += blockDim.x) {
      if (shared_scores[e] > my_val) {
        my_val = shared_scores[e];
        my_expert = e;
      }
    }

    float warp_max_val = my_val;
    int warp_max_expert = my_expert;

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, warp_max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, warp_max_expert, offset);
      if (other_val > warp_max_val) {
        warp_max_val = other_val;
        warp_max_expert = other_expert;
      }
    }

    if (lane_id == 0 && warp_id < kWarpsPerToken) {
      warp_maxs[warp_id] = warp_max_val;
      warp_experts[warp_id] = warp_max_expert;
    }

    __syncthreads();

    if (warp_id == 0) {
      float final_max = (lane_id < kWarpsPerToken) ? warp_maxs[lane_id] : -FLT_MAX;
      int final_expert = (lane_id < kWarpsPerToken) ? warp_experts[lane_id] : -1;

#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, final_max, offset);
        int other_expert = __shfl_down_sync(0xFFFFFFFF, final_expert, offset);
        if (other_val > final_max) {
          final_max = other_val;
          final_expert = other_expert;
        }
      }

      if (lane_id == 0) {
        selected_experts[k] = final_expert;
      }
    }

    __syncthreads();

    int selected = selected_experts[k];
    if (selected >= 0 && tid == 0) {
      shared_scores[selected] = -FLT_MAX;
    }

    __syncthreads();
  }

  static_assert(kMaxTopK <= device::kWarpThreads);
  if (tid >= device::kWarpThreads) return;

  // only use the first warp to perform write to global operation
  float routed_weight = 0.0f;
  int32_t selected_expert = 0;
  if (tid < topk_routed) {
    int expert_id = selected_experts[tid];
    float score = shared_original_scores[expert_id];
    if (expert_id >= 0 && expert_id < static_cast<int>(num_experts)) {
      routed_weight = score;
      selected_expert = expert_id;
    }
  }
  const auto routed_sum = device::warp::reduce_sum<kMaxTopK>(routed_weight);
  if (tid < topk) {
    const bool is_shared = tid >= topk_routed;
    const auto output_offset = row_idx * topk + tid;
    const auto weight = is_shared ? (routed_sum / routed_scaling_factor) : routed_weight;
    const auto expert_id = is_shared ? (num_experts + tid - topk_routed) : selected_expert;
    const auto scale = apply_routed_scaling_factor_on_output ? routed_scaling_factor : 1.0f;
    const auto norm = renormalize && routed_sum > 0.0f ? routed_sum : 1.0f;
    output[output_offset] = weight / norm * scale;
    indices[output_offset] = expert_id;
  }
}

template <ScoringFunc kScoringFunc>
__global__ void moe_fused_gate_kernel(const MoEFusedGateParams __grid_constant__ params) {
  const auto& [input, bias, output, indices, num_rows, num_experts, topk, num_fused_shared_experts, renormalize, routed_scaling_factor, apply_routed_scaling_factor_on_output] =
      params;

  uint32_t row_idx = blockIdx.x * kWarpsPerCTA + threadIdx.y;
  if (row_idx >= num_rows) return;

  // number of routed experts to select (excluding fused shared experts)
  const uint32_t topk_routed = topk - num_fused_shared_experts;

  uint32_t lane_id = threadIdx.x;
  uint32_t warp_id = threadIdx.y;

  extern __shared__ float shared_mem[];
  float* shared_scores = shared_mem + warp_id * num_experts * 2;
  float* shared_original_scores = shared_scores + num_experts;
  __shared__ int selected_experts[kWarpsPerCTA][kMaxTopK];
  int* warp_selected_experts = selected_experts[warp_id];

  for (uint32_t e = lane_id; e < num_experts; e += kWarpSize) {
    float input_val = input[row_idx * num_experts + e];
    float bias_val = bias[e];
    float score_val = compute_score<kScoringFunc>(input_val);
    float biased_val = score_val + bias_val;
    shared_scores[e] = biased_val;
    shared_original_scores[e] = score_val;
  }

  __syncwarp();

  // only select topk_routed experts
  for (uint32_t k = 0; k < topk_routed; k++) {
    float max_val = -FLT_MAX;
    int max_expert = -1;

    for (uint32_t expert = lane_id; expert < num_experts; expert += kWarpSize) {
      if (shared_scores[expert] > max_val) {
        max_val = shared_scores[expert];
        max_expert = expert;
      }
    }

    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
      int other_expert = __shfl_down_sync(0xFFFFFFFF, max_expert, offset);

      if (other_val > max_val || (other_val == max_val && other_expert < max_expert)) {
        max_val = other_val;
        max_expert = other_expert;
      }
    }

    if (lane_id == 0) {
      warp_selected_experts[k] = max_expert;
      if (max_expert != -1) {
        shared_scores[max_expert] = -FLT_MAX;
      }
    }

    __syncwarp();
  }

  static_assert(kMaxTopK <= device::kWarpThreads);

  float routed_weight = 0.0f;
  int32_t selected_expert = 0;
  if (lane_id < topk_routed) {
    int expert_id = warp_selected_experts[lane_id];
    if (expert_id >= 0 && expert_id < static_cast<int>(num_experts)) {
      routed_weight = shared_original_scores[expert_id];
      selected_expert = expert_id;
    }
  }
  const auto routed_sum = device::warp::reduce_sum<kMaxTopK>(routed_weight);
  if (lane_id < topk) {
    const bool is_shared = lane_id >= topk_routed;
    const auto output_idx = row_idx * topk + lane_id;
    const auto weight = is_shared ? (routed_sum / routed_scaling_factor) : routed_weight;
    const auto expert_id = is_shared ? (num_experts + lane_id - topk_routed) : selected_expert;
    const auto scale = apply_routed_scaling_factor_on_output ? routed_scaling_factor : 1.0f;
    const auto norm = renormalize && routed_sum > 0.0f ? routed_sum : 1.0f;
    output[output_idx] = weight / norm * scale;
    indices[output_idx] = expert_id;
  }
}

template <ScoringFunc kScoringFunc>
void dispatch_small_token_kernel(
    uint32_t num_rows,
    uint32_t threads_per_block,
    uint32_t warps_per_token,
    DLDevice device,
    size_t smem_per_row,
    const MoEFusedGateParams& params) {
  using namespace host;
  if (warps_per_token <= 8) {
    LaunchKernel(num_rows, threads_per_block, device, smem_per_row)(
        moe_fused_gate_kernel_small_token<8, kScoringFunc>, params);
  } else if (warps_per_token <= 12) {
    LaunchKernel(num_rows, threads_per_block, device, smem_per_row)(
        moe_fused_gate_kernel_small_token<12, kScoringFunc>, params);
  } else {
    LaunchKernel(num_rows, threads_per_block, device, smem_per_row)(
        moe_fused_gate_kernel_small_token<16, kScoringFunc>, params);
  }
}

// Recovered baseline launch path for off-domain configurations. Bit-identical to
// the baseline's MoEFusedGateKernel::run body (shape-based small/large dispatch).
inline void launch(
    uint32_t num_rows,
    uint32_t num_experts,
    uint32_t scoring_func,
    DLDevice device,
    const MoEFusedGateParams& params) {
  using namespace host;

  const size_t smem_per_row = 2 * num_experts * sizeof(float);
  bool use_small_token_kernel = num_rows <= kSmallTokenThreshold;

  if (use_small_token_kernel) {
    // 1 token per block
    uint32_t warps_per_token = div_ceil(num_experts, kWarpSize);
    warps_per_token = std::min(warps_per_token, 16u);
    uint32_t threads_per_block = warps_per_token * kWarpSize;

    if (scoring_func == 0) {
      dispatch_small_token_kernel<ScoringFunc::kSigmoid>(
          num_rows, threads_per_block, warps_per_token, device, smem_per_row, params);
    } else {
      dispatch_small_token_kernel<ScoringFunc::kSqrtSoftplus>(
          num_rows, threads_per_block, warps_per_token, device, smem_per_row, params);
    }
  } else {
    // multiple tokens per block
    uint32_t num_blocks = div_ceil(num_rows, kWarpsPerCTA);
    dim3 block_dim(kWarpSize, kWarpsPerCTA);
    size_t large_smem = smem_per_row * kWarpsPerCTA;

    if (scoring_func == 0) {
      LaunchKernel(num_blocks, block_dim, device, large_smem)(
          moe_fused_gate_kernel<ScoringFunc::kSigmoid>, params);
    } else {
      LaunchKernel(num_blocks, block_dim, device, large_smem)(
          moe_fused_gate_kernel<ScoringFunc::kSqrtSoftplus>, params);
    }
  }
}

}  // namespace fallback

// ─────────────────────────────────────────────────────────────────────────────
// Cold-safe candidate kernel: one warp per token, WARPS_PER_BLOCK tokens/block.
// Uses NO shared memory — all reductions are intra-warp shuffles over the full
// 32-lane warp, so there is ZERO uninitialized read on any (including cold) launch.
//
// Each of the 32 lanes owns kExpertsPerLane experts {lane, lane+32, ...} and keeps
// their (un-biased score, biased score) pairs in registers. Selection is the same
// iterative descending-biased arg-max with a smaller-index tie-break as the
// baseline; the emitted weight uses the un-biased score; the routed-score sum,
// renorm and scaling reproduce the baseline's exact float32 op order.
// ─────────────────────────────────────────────────────────────────────────────
constexpr uint32_t kCandWarpSize = 32;
constexpr uint32_t kCandMaxTopK = 16;  // reduce_sum width, matches the baseline

// Monotonic float->uint key so an unsigned 64-bit max selects the larger float and,
// on ties, the SMALLER expert index (idx stored as 65535 - idx in the low bits, so a
// larger packed value == a smaller index). Expert indices are < 65535, so the index
// field never collides with a neighbouring value bucket.
__device__ __forceinline__ uint64_t pack_val_idx(float val, int32_t idx) {
  uint32_t val_bits = __float_as_uint(val);
  val_bits ^= (val_bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  uint32_t idx_bits = static_cast<uint32_t>(65535 - idx);
  return (static_cast<uint64_t>(val_bits) << 32) | idx_bits;
}

__device__ __forceinline__ void unpack_val_idx(uint64_t packed, float& val, int32_t& idx) {
  uint32_t idx_bits = static_cast<uint32_t>(packed & 0xFFFFFFFFu);
  idx = static_cast<int32_t>(65535 - static_cast<int32_t>(idx_bits));
  uint32_t val_bits = static_cast<uint32_t>(packed >> 32);
  val_bits ^= (val_bits & 0x80000000u) ? 0x80000000u : 0xffffffffu;
  val = __uint_as_float(val_bits);
}

__device__ __forceinline__ uint64_t warp_max_u64(uint64_t val) {
#pragma unroll
  for (int mask = kCandWarpSize / 2; mask > 0; mask >>= 1) {
    uint64_t other = __shfl_xor_sync(0xffffffffu, val, mask);
    val = max(val, other);
  }
  return val;
}

template <uint32_t kWarpsPerBlock, uint32_t kExpertsPerLane, ScoringFunc kScoringFunc>
__global__ void moe_fused_gate_warp_per_token_kernel(const fallback::MoEFusedGateParams __grid_constant__ params) {
  const float* __restrict__ input = params.input;
  const float* __restrict__ bias = params.bias;
  float* __restrict__ output = params.output;
  int32_t* __restrict__ indices = params.indices;
  const uint32_t num_rows = params.num_rows;
  const uint32_t num_experts = params.num_experts;
  const uint32_t topk = params.topk;
  const uint32_t num_fused_shared_experts = params.num_fused_shared_experts;
  const bool renormalize = params.renormalize;
  const float routed_scaling_factor = params.routed_scaling_factor;
  const bool apply_routed_scaling_factor_on_output = params.apply_routed_scaling_factor_on_output;

  const uint32_t lane_id = threadIdx.x;
  const uint32_t warp_in_block = threadIdx.y;
  const uint32_t row_idx = blockIdx.x * kWarpsPerBlock + warp_in_block;
  // row_idx depends only on block/warp indices, identical across all 32 lanes, so
  // the whole warp returns together — every surviving shuffle has all lanes active.
  if (row_idx >= num_rows) return;

  const uint32_t topk_routed = topk - num_fused_shared_experts;

  const float* __restrict__ token_input = input + static_cast<size_t>(row_idx) * num_experts;

  // Per-lane register-resident expert slots. Out-of-range experts are neutral:
  // score 0 and biased -FLT_MAX (never selected, never contribute to the sum).
  float sig[kExpertsPerLane];
  float biased[kExpertsPerLane];
#pragma unroll
  for (uint32_t j = 0; j < kExpertsPerLane; ++j) {
    const uint32_t e = lane_id + j * kCandWarpSize;
    if (e < num_experts) {
      const float s = compute_score<kScoringFunc>(token_input[e]);
      sig[j] = s;
      biased[j] = s + bias[e];
    } else {
      sig[j] = 0.0f;
      biased[j] = -FLT_MAX;
    }
  }

  // Iterative descending-biased arg-max. After slot k is decided its winner lives
  // in lane k (winner_idx / winner_sig), matching the baseline's per-lane layout so
  // the routed-sum reduction below is the identical butterfly over identical lanes.
  int32_t winner_idx = 0;
  float winner_sig = 0.0f;
  for (uint32_t k = 0; k < topk_routed; ++k) {
    float local_val = -FLT_MAX;
    int32_t local_idx = static_cast<int32_t>(lane_id);
#pragma unroll
    for (uint32_t j = 0; j < kExpertsPerLane; ++j) {
      if (biased[j] > local_val) {
        local_val = biased[j];
        local_idx = static_cast<int32_t>(lane_id + j * kCandWarpSize);
      }
    }
    const uint64_t best = warp_max_u64(pack_val_idx(local_val, local_idx));
    float best_val;
    int32_t best_idx;
    unpack_val_idx(best, best_val, best_idx);

    const uint32_t owner = static_cast<uint32_t>(best_idx) & (kCandWarpSize - 1);
    float owned_sig = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kExpertsPerLane; ++j) {
      if (static_cast<int32_t>(lane_id + j * kCandWarpSize) == best_idx) owned_sig = sig[j];
    }
    const float sel_sig = __shfl_sync(0xffffffffu, owned_sig, owner);

    if (lane_id == k) {
      winner_idx = best_idx;
      winner_sig = sel_sig;
    }
    // Mask the chosen expert on its owning lane so it cannot be picked again.
#pragma unroll
    for (uint32_t j = 0; j < kExpertsPerLane; ++j) {
      if (lane_id == owner && static_cast<int32_t>(lane_id + j * kCandWarpSize) == best_idx)
        biased[j] = -FLT_MAX;
    }
    __syncwarp();
  }

  // routed_weight holds the selected un-biased score in lanes [0, topk_routed) and
  // 0 elsewhere — identical to the baseline before reduce_sum<kMaxTopK>.
  const float routed_weight = (lane_id < topk_routed) ? winner_sig : 0.0f;
  const float routed_sum = device::warp::reduce_sum<kCandMaxTopK>(routed_weight);

  if (lane_id < topk) {
    const bool is_shared = lane_id >= topk_routed;
    const auto output_offset = static_cast<size_t>(row_idx) * topk + lane_id;
    const float weight = is_shared ? (routed_sum / routed_scaling_factor) : winner_sig;
    const int32_t expert_id =
        is_shared ? static_cast<int32_t>(num_experts + lane_id - topk_routed) : winner_idx;
    const float scale = apply_routed_scaling_factor_on_output ? routed_scaling_factor : 1.0f;
    const float norm = (renormalize && routed_sum > 0.0f) ? routed_sum : 1.0f;
    output[output_offset] = weight / norm * scale;
    indices[output_offset] = expert_id;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exported entry point. Identical ABI / validation to the baseline.
// ─────────────────────────────────────────────────────────────────────────────
struct MoEFusedGateKernel {
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView indices,
      uint32_t topk,
      uint32_t scoring_func,  // 0 = sigmoid, 1 = sqrtsoftplus
      uint32_t num_fused_shared_experts,
      bool renormalize,
      float routed_scaling_factor,
      bool apply_routed_scaling_factor_on_output) {
    using namespace host;

    auto N = SymbolicSize{"num_rows"};
    auto E = SymbolicSize{"num_experts"};
    auto K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    K.set_value(topk);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, E}).with_dtype<float>().with_device(device).verify(input);
    TensorMatcher({E}).with_dtype<float>().with_device(device).verify(bias);
    TensorMatcher({N, K}).with_dtype<float>().with_device(device).verify(output);
    TensorMatcher({N, K}).with_dtype<int32_t>().with_device(device).verify(indices);

    const auto num_rows = static_cast<uint32_t>(N.unwrap());
    const auto num_experts = static_cast<uint32_t>(E.unwrap());

    RuntimeCheck(num_experts <= fallback::kMaxExperts, "num_experts exceeds maximum supported value");
    RuntimeCheck(scoring_func <= 1, "scoring_func must be 0 (sigmoid) or 1 (sqrtsoftplus)");
    RuntimeCheck(topk > num_fused_shared_experts, "topk must be greater than num_fused_shared_experts");

    const auto params = fallback::MoEFusedGateParams{
        .input = static_cast<const float*>(input.data_ptr()),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .output = static_cast<float*>(output.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .num_rows = num_rows,
        .num_experts = num_experts,
        .topk = topk,
        .num_fused_shared_experts = num_fused_shared_experts,
        .renormalize = renormalize,
        .routed_scaling_factor = routed_scaling_factor,
        .apply_routed_scaling_factor_on_output = apply_routed_scaling_factor_on_output,
    };

    if (num_rows == 0) return;

    // ── Dispatch gate ─────────────────────────────────────────────────────────
    // Host-side production-domain predicate. In-domain -> cold-safe warp-per-token
    // candidate kernel; everything else -> recovered baseline (bit-identical).
    const bool in_domain =
        (num_experts == 128 && topk == 5 && scoring_func == 0 && num_fused_shared_experts == 1 &&
         renormalize && routed_scaling_factor == 2.0f && apply_routed_scaling_factor_on_output);

    if (in_domain) {
      // E=128 -> 4 experts per lane over a 32-lane warp; 8 tokens per block.
      constexpr uint32_t kWarpsPerBlock = 8;
      constexpr uint32_t kExpertsPerLane = 4;  // div_ceil(128, 32)
      const uint32_t num_blocks = div_ceil(num_rows, kWarpsPerBlock);
      dim3 block_dim(kCandWarpSize, kWarpsPerBlock);
      LaunchKernel(num_blocks, block_dim, device.unwrap())(
          moe_fused_gate_warp_per_token_kernel<kWarpsPerBlock, kExpertsPerLane, ScoringFunc::kSigmoid>,
          params);
    } else {
      // Off-domain: recovered baseline behaviour, bit-identical.
      fallback::launch(num_rows, num_experts, scoring_func, device.unwrap(), params);
    }
  }
};

}  // namespace
