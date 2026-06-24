// Native-CUDA candidate for sgl_kernel.fp8_scaled_mm, same destination-passing
// TVM-FFI ABI as the baseline, compiled together in one module (symmetric flags).
//
// Fast path (this round): a bandwidth-optimal M==1 FP8 GEMV for the decode
// regime. The recovered sm100 baseline uses a 64-row CUTLASS MMA tile even for
// M=1 (measured ~8.6% of HBM peak); this kernel instead streams B once with
// fully-coalesced vector loads.
//
// Layout exploited: B is column-major [K,N] (stride (1,K)), i.e. physically an
// [N,K] contiguous tensor Bphys with Bphys[n,k] == B[k,n]. So each output column
// n maps to a CONTIGUOUS K-length row Bphys[n,:]. One warp owns one column n and
// reads that row in 16-byte (uint4 = 16xfp8) chunks; A[0,:] is preloaded to
// shared and reused by all warps. Per-lane fp32 accumulation -> warp-shuffle
// reduce -> lane 0 applies scale_a[0]*scale_b[n], stores bf16.
//
// Everything outside the covered regime (M!=1, non-column-major B, non-bf16 out,
// non-fp8 inputs, K not a multiple of 16, bias, bad scales) falls back to the
// recovered baseline so correctness is never lost. The route diagnostic exposes
// which path fires. (Small-M swap-AB and small-N split-K are follow-ups.)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/all.h>

#include "fp8_scaled_mm_abi.h"
#include "fp8_swapab_smallm.h"

namespace {

constexpr int kWarps = 8;        // warps per CTA (one output column each)
constexpr int kBlock = kWarps * 32;
constexpr int kVec = 16;         // fp8 per uint4 load

// One warp per output column n. A[0,:] (K fp8) lives in shared, reused by all warps.
__global__ void fp8_gemv_m1_kernel(
    const __nv_fp8_e4m3* __restrict__ A,    // [K]            (row of A; M==1)
    const __nv_fp8_e4m3* __restrict__ B,    // [N, K] (Bphys; B[k,n] = B[n*K + k])
    const float* __restrict__ scale_a,      // [1]
    const float* __restrict__ scale_b,      // [N]
    __nv_bfloat16* __restrict__ out,        // [N]
    int K, int N) {
  extern __shared__ __align__(16) char smem_raw[];
  __nv_fp8_e4m3* sA = reinterpret_cast<__nv_fp8_e4m3*>(smem_raw);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Cooperative vectorized preload of A[0,:] into shared (K % 16 == 0 guaranteed
  // by the dispatch predicate, so uint4 copies are safe and aligned).
  for (int i = tid * kVec; i < K; i += kBlock * kVec) {
    *reinterpret_cast<uint4*>(&sA[i]) = *reinterpret_cast<const uint4*>(&A[i]);
  }
  __syncthreads();

  const int n = blockIdx.x * kWarps + warp;
  if (n >= N) return;

  const __nv_fp8_e4m3* Brow = B + static_cast<size_t>(n) * K;
  float acc = 0.f;
  // Lanes cover consecutive 16-fp8 chunks -> 512 contiguous B bytes / warp / step.
  for (int k = lane * kVec; k < K; k += 32 * kVec) {
    uint4 bvec = *reinterpret_cast<const uint4*>(Brow + k);
    uint4 avec = *reinterpret_cast<const uint4*>(&sA[k]);
    const __nv_fp8_e4m3* bb = reinterpret_cast<const __nv_fp8_e4m3*>(&bvec);
    const __nv_fp8_e4m3* aa = reinterpret_cast<const __nv_fp8_e4m3*>(&avec);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      acc += static_cast<float>(bb[j]) * static_cast<float>(aa[j]);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) {
    out[n] = __float2bfloat16(acc * scale_a[0] * scale_b[n]);
  }
}

template <typename T>
inline const T* cptr(const tvm::ffi::TensorView& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data_ptr()) + t.byte_offset());
}
template <typename T>
inline T* mptr(const tvm::ffi::TensorView& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data_ptr()) + t.byte_offset());
}
// Exact fp8_e4m3fn (DLPack code 10 == kDLFloat8_e4m3fn), NOT just any 8-bit type:
// uint8 (kDLUInt=1), int8 (kDLInt=0), and fp8_e5m2 (kDLFloat8_e5m2=12) must NOT
// reach the kernel, which reinterprets bytes as __nv_fp8_e4m3.
inline bool is_f8e4m3(const tvm::ffi::TensorView& t) {
  return t.dtype().code == kDLFloat8_e4m3fn && t.dtype().bits == 8 && t.dtype().lanes == 1;
}
inline bool is_f32(const tvm::ffi::TensorView& t) {
  return t.dtype().code == kDLFloat && t.dtype().bits == 32 && t.dtype().lanes == 1;
}
inline bool is_bf16(const tvm::ffi::TensorView& t) {
  return t.dtype().code == kDLBfloat && t.dtype().bits == 16 && t.dtype().lanes == 1;
}
// Contiguous 2-D [d0,d1] with stride(0)==d1 (so flat [i] indexing on dim 0 works;
// the scales are [M,1]/[N,1] and the kernel reads scale_b[n] from offset n).
inline bool is_contig_2d(const tvm::ffi::TensorView& t, int64_t d0, int64_t d1) {
  return t.ndim() == 2 && t.size(0) == d0 && t.size(1) == d1 && t.stride(0) == d1;
}
inline bool same_cuda_dev(const tvm::ffi::TensorView& ref, const tvm::ffi::TensorView& t) {
  return t.device().device_type == kDLCUDA && t.device().device_type == ref.device().device_type &&
         t.device().device_id == ref.device().device_id;
}

// Coverage predicate (cheap; no launch, no sync): the M==1 GEMV fast path.
// Strict: any input outside the exact covered contract returns false -> baseline
// fallback. Validates dtype CODE (not just width), 2-D shapes, A row-major /
// B column-major / out row-major layout, scale rank+shape+contiguity, and that
// all tensors live on one CUDA device.
inline bool covers_m1_gemv(
    const tvm::ffi::TensorView& a, const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a, const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::TensorView& out) {
  if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2) return false;
  const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  if (M != 1) return false;                        // M==1 only this round
  // Measured on B200 (round 0): the scalar-fp8-decode GEMV is instruction-bound
  // for the largest-work shapes, where the baseline's tensor-core tiling wins
  // (k8192_n3072 0.86x, k8192_n4608 0.73x). Fall back when BOTH K and N are
  // large so the fast path never regresses a covered shape; all measured
  // winners have K<4096 or N<3072. (A faster-decode / tensor-core path is the
  // follow-up to widen this.)
  if (K >= 4096 && N >= 3072) return false;
  // dtypes (exact codes)
  if (!is_f8e4m3(a) || !is_f8e4m3(b) || !is_bf16(out) || !is_f32(scales_a) || !is_f32(scales_b))
    return false;
  // shapes / layout
  if (b.size(0) != K) return false;
  if (a.stride(1) != 1) return false;              // A row-major
  if (b.stride(0) != 1) return false;              // B column-major (Bphys [N,K])
  if (b.stride(1) != K) return false;              // exactly-packed Bphys: the GEMV addresses B + n*K, so the leading dim must be K (rejects padded/sliced column-major B)
  if (K % kVec != 0) return false;                 // vectorized 16-fp8 loads
  if (!(out.size(0) == M && out.size(1) == N && out.stride(1) == 1)) return false;  // out row-major [M,N]
  if ((N * 2) % 16 != 0) return false;             // bf16 out row must be 16-byte aligned (N%8==0), matching the baseline contract
  if (!is_contig_2d(scales_a, M, 1)) return false; // scale_a [M,1] contiguous
  if (!is_contig_2d(scales_b, N, 1)) return false; // scale_b [N,1] contiguous
  // device consistency: all tensors on the same CUDA device
  if (!same_cuda_dev(a, b) || !same_cuda_dev(a, scales_a) ||
      !same_cuda_dev(a, scales_b) || !same_cuda_dev(a, out))
    return false;
  return true;
}

// Small-M swap-AB is DISABLED (Round 2 evidence-backed no-go; see the predicate
// body). Flip to true to re-route small-M to the swap-AB kernel.
inline constexpr bool kSmallMSwapAbEnabled = false;

// Coverage predicate for the small-M swap-AB path (M in (1,64]). Same strict
// contract as the M=1 GEMV (exact dtypes, A row-major / B column-major / out
// row-major, scale rank+contiguity, same CUDA device, K%16==0).
inline bool covers_smallm_swapab(
    const tvm::ffi::TensorView& a, const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a, const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::TensorView& out) {
  // EVIDENCE-BACKED NO-GO (Round 2): the swap-AB kernel (solution/fp8_swapab_smallm.cu)
  // is implemented and correct but LOSES on every measured small-M shape (geomean
  // 0.85x vs baseline; NCU: tensor-core activity ~1.9%, occupancy ~11% — the tiny
  // swapped-N dimension (=M) under-fills the 2-SM warp-specialized mainloop). The
  // baseline Gemm64 is already 36-89% M-tile-utilized for M=23-57. So small-M falls
  // back to the baseline (no regression). The predicate logic + kernel stay in tree
  // as the documented attempt; flip kSmallMSwapAbEnabled to re-enable if a future
  // tile/pipeline tuning wins. See docs/results.md (small-M no-go).
  if (!kSmallMSwapAbEnabled) return false;
  if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2) return false;
  const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  if (M < 2 || M > 64) return false;               // small-M regime
  if (!is_f8e4m3(a) || !is_f8e4m3(b) || !is_bf16(out) || !is_f32(scales_a) || !is_f32(scales_b))
    return false;
  if (b.size(0) != K) return false;
  if (a.stride(1) != 1) return false;              // A row-major
  if (b.stride(0) != 1) return false;              // B column-major
  if (b.stride(1) != K) return false;              // exactly-packed Bphys (swap-AB uses a packed [N,K] stride)
  if (K % kVec != 0) return false;                 // 16-fp8 alignment along K
  if (!(out.size(0) == M && out.size(1) == N && out.stride(1) == 1)) return false;
  if ((N * 2) % 16 != 0) return false;             // bf16 out row must be 16-byte aligned (N%8==0)
  if (!is_contig_2d(scales_a, M, 1)) return false;
  if (!is_contig_2d(scales_b, N, 1)) return false;
  if (!same_cuda_dev(a, b) || !same_cuda_dev(a, scales_a) ||
      !same_cuda_dev(a, scales_b) || !same_cuda_dev(a, out))
    return false;
  return true;
}

inline void fallback(
    tvm::ffi::TensorView a, tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a, tvm::ffi::TensorView scales_b, tvm::ffi::TensorView out) {
  // Validate dtypes at the TensorView boundary before the forced-dtype view, so
  // a wrong-dtype input (e.g. e5m2/uint8) is rejected, not reinterpreted.
  fp8abi::require_fp8_contract(a, b, scales_a, scales_b, out);
  auto out_dtype = (out.dtype().code == kDLBfloat) ? torch::kBFloat16 : torch::kHalf;
  auto ta = fp8abi::view_as(a, torch::kFloat8_e4m3fn);
  auto tb = fp8abi::view_as(b, torch::kFloat8_e4m3fn);
  auto tsa = fp8abi::view_as(scales_a, torch::kFloat32);
  auto tsb = fp8abi::view_as(scales_b, torch::kFloat32);
  auto tout = fp8abi::view_as(out, out_dtype);
  fp8_scaled_mm_baseline_impl(tout, ta, tb, tsa, tsb, c10::nullopt);
}

}  // namespace

void fp8_scaled_mm_candidate(
    tvm::ffi::TensorView a, tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a, tvm::ffi::TensorView scales_b, tvm::ffi::TensorView out) {
  if (covers_m1_gemv(a, b, scales_a, scales_b, out)) {
    // Pin the current device to the tensors' device so getCurrentCUDAStream() and
    // the launch target the right GPU even if the caller's current device differs
    // (covers_m1_gemv already requires all tensors share this CUDA device).
    const c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(a.device().device_id));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));
    const int grid = (N + kWarps - 1) / kWarps;
    const size_t smem = static_cast<size_t>(K) * sizeof(__nv_fp8_e4m3);
    auto stream = at::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(a.device().device_id));
    fp8_gemv_m1_kernel<<<grid, kBlock, smem, stream>>>(
        cptr<__nv_fp8_e4m3>(a), cptr<__nv_fp8_e4m3>(b),
        cptr<float>(scales_a), cptr<float>(scales_b),
        mptr<__nv_bfloat16>(out), K, N);
    return;
  }
  if (covers_smallm_swapab(a, b, scales_a, scales_b, out)) {
    auto ta = fp8abi::view_as(a, torch::kFloat8_e4m3fn);
    auto tb = fp8abi::view_as(b, torch::kFloat8_e4m3fn);
    auto tsa = fp8abi::view_as(scales_a, torch::kFloat32);
    auto tsb = fp8abi::view_as(scales_b, torch::kFloat32);
    auto tout = fp8abi::view_as(out, torch::kBFloat16);
    fp8_scaled_mm_swapab_smallm(tout, ta, tb, tsa, tsb);
    return;
  }
  fallback(a, b, scales_a, scales_b, out);
}

// Route diagnostic: 1 = M=1 GEMV fast path, 2 = small-M swap-AB fast path,
// 0 = baseline fallback. Launches nothing.
int64_t fp8_scaled_mm_candidate_route(
    tvm::ffi::TensorView a, tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a, tvm::ffi::TensorView scales_b, tvm::ffi::TensorView out) {
  if (covers_m1_gemv(a, b, scales_a, scales_b, out)) return 1;
  if (covers_smallm_swapab(a, b, scales_a, scales_b, out)) return 2;
  return 0;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_scaled_mm_candidate, fp8_scaled_mm_candidate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_scaled_mm_candidate_route, fp8_scaled_mm_candidate_route);

// --- Test-only bias-capable candidate fallback (AC-3.1 `bias!=None` edge) ---
// The captured production workload is 100% bias=None, so the main candidate ABI
// carries no bias. This 6-arg entry proves the required safety property: a biased
// call on an otherwise-covered shape does NOT enter the bias-unaware M=1 GEMV /
// swap-AB fast paths — it routes to the recovered baseline (route 0) and is
// numerically correct (out = (A@B)*scale_a*scale_b + bias). It is exercised by
// bench/correctness.py:bias_edge_test().
void fp8_scaled_mm_candidate_bias(
    tvm::ffi::TensorView a, tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a, tvm::ffi::TensorView scales_b,
    tvm::ffi::TensorView bias, tvm::ffi::TensorView out) {
  fp8abi::require_fp8_contract(a, b, scales_a, scales_b, out);
  const int64_t N = b.size(1);
  int64_t bias_numel = 1;
  for (int i = 0; i < bias.ndim(); ++i) bias_numel *= bias.size(i);
  TORCH_CHECK(bias_numel == N, "fp8_scaled_mm: bias numel must equal N");
  TORCH_CHECK(bias.device().device_type == kDLCUDA &&
                  bias.device().device_id == a.device().device_id,
              "fp8_scaled_mm: bias must be on the same CUDA device");
  const bool out_bf16 = (out.dtype().code == kDLBfloat);
  auto out_dtype = out_bf16 ? torch::kBFloat16 : torch::kHalf;
  TORCH_CHECK(bias.dtype().bits == 16 && bias.dtype().code == out.dtype().code,
              "fp8_scaled_mm: bias dtype must match out_dtype");
  // Bias-unaware fast paths are skipped entirely: unconditional baseline fallback.
  auto ta = fp8abi::view_as(a, torch::kFloat8_e4m3fn);
  auto tb = fp8abi::view_as(b, torch::kFloat8_e4m3fn);
  auto tsa = fp8abi::view_as(scales_a, torch::kFloat32);
  auto tsb = fp8abi::view_as(scales_b, torch::kFloat32);
  auto tbias = fp8abi::view_as(bias, out_dtype);
  auto tout = fp8abi::view_as(out, out_dtype);
  fp8_scaled_mm_baseline_impl(tout, ta, tb, tsa, tsb, tbias);
}

// Route for a biased call: always 0 (no fast path is bias-aware -> baseline).
int64_t fp8_scaled_mm_candidate_bias_route(
    tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView,
    tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView) {
  return 0;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_scaled_mm_candidate_bias, fp8_scaled_mm_candidate_bias);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_scaled_mm_candidate_bias_route, fp8_scaled_mm_candidate_bias_route);
