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
inline bool is_f8e4m3(const tvm::ffi::TensorView& t) {
  // DLPack float8_e4m3 code; accept by 8-bit width (torch float8_e4m3fn).
  return t.dtype().bits == 8 && t.dtype().lanes == 1;
}

// Coverage predicate (cheap; no launch, no sync): the M==1 GEMV fast path.
inline bool covers_m1_gemv(
    const tvm::ffi::TensorView& a, const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a, const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::TensorView& out) {
  if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2) return false;
  const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  if (M != 1) return false;                       // M==1 only this round
  // Measured on B200 (round 0): the scalar-fp8-decode GEMV is instruction-bound
  // for the largest-work shapes, where the baseline's tensor-core tiling wins
  // (k8192_n3072 0.86x, k8192_n4608 0.73x). Fall back when BOTH K and N are
  // large so the fast path never regresses a covered shape; all measured
  // winners have K<4096 or N<3072. (A faster-decode / tensor-core path is the
  // follow-up to widen this.)
  if (K >= 4096 && N >= 3072) return false;
  if (b.size(0) != K) return false;
  if (b.stride(0) != 1) return false;             // column-major B (Bphys [N,K])
  if (a.stride(1) != 1) return false;             // row-major A
  if (K % kVec != 0) return false;                // vectorized loads
  if (out.dtype().code != kDLBfloat || out.dtype().bits != 16) return false;  // bf16 out
  if (!is_f8e4m3(a) || !is_f8e4m3(b)) return false;
  if (scales_a.dtype().code != kDLFloat || scales_a.dtype().bits != 32) return false;
  if (scales_b.dtype().code != kDLFloat || scales_b.dtype().bits != 32) return false;
  if (scales_a.size(0) != M || scales_b.size(0) != N) return false;
  return true;
}

inline void fallback(
    tvm::ffi::TensorView a, tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a, tvm::ffi::TensorView scales_b, tvm::ffi::TensorView out) {
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
  if (!covers_m1_gemv(a, b, scales_a, scales_b, out)) {
    fallback(a, b, scales_a, scales_b, out);
    return;
  }
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));
  const int grid = (N + kWarps - 1) / kWarps;
  const size_t smem = static_cast<size_t>(K) * sizeof(__nv_fp8_e4m3);
  auto stream = at::cuda::getCurrentCUDAStream();
  fp8_gemv_m1_kernel<<<grid, kBlock, smem, stream>>>(
      cptr<__nv_fp8_e4m3>(a), cptr<__nv_fp8_e4m3>(b),
      cptr<float>(scales_a), cptr<float>(scales_b),
      mptr<__nv_bfloat16>(out), K, N);
}

int64_t fp8_scaled_mm_candidate_route(
    tvm::ffi::TensorView a, tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a, tvm::ffi::TensorView scales_b, tvm::ffi::TensorView out) {
  return covers_m1_gemv(a, b, scales_a, scales_b, out) ? 1 : 0;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_scaled_mm_candidate, fp8_scaled_mm_candidate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_scaled_mm_candidate_route, fp8_scaled_mm_candidate_route);
