// Native-CUDA candidate for sgl_kernel.fp8_scaled_mm, exposed through the same
// destination-passing TVM-FFI ABI as the baseline. Compiled together with the
// baseline in one module (symmetric flags).
//
// STATUS (Round 0 scaffold): identity stub — every shape currently falls back to
// the recovered baseline via fp8_scaled_mm_baseline_impl(), so the harness is
// green before any kernel is written (correctness == baseline, speedup ~1.0).
// The decode-regime specialized kernel (swap-AB / skinny-GEMV for M<=64 on the
// captured column-major B) replaces the fast path below; uncovered shapes keep
// falling back so correctness is never lost.
//
// This TU does NOT include the verbatim baseline source; it calls the extern
// fp8_scaled_mm_baseline_impl() declared in fp8_scaled_mm_abi.h, so the upstream
// fp8_gemm_kernel.cu is compiled exactly once (in the baseline TU).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "fp8_scaled_mm_abi.h"

namespace {

// Predicate: does the candidate have a specialized fast path for these inputs?
// Round 0: no specialized kernel yet -> always false (pure baseline fallback).
// When the decode kernel lands this returns true for the covered regime
// (e.g. M<=64, column-major B, bf16 out, bias=None, fp8e4m3 A/B) and false
// otherwise. Cheap: shape/dtype/stride inspection only, no host sync, no launch.
inline bool candidate_covers(
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& b,
    const tvm::ffi::TensorView& scales_a,
    const tvm::ffi::TensorView& scales_b,
    const tvm::ffi::TensorView& out) {
  return false;  // scaffold: fall back everywhere
}

}  // namespace

// Destination-passing candidate. Covered shapes -> specialized kernel (none yet);
// everything else -> recovered baseline. Output pre-allocated, passed last.
void fp8_scaled_mm_candidate(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::TensorView out) {
  if (!candidate_covers(a, b, scales_a, scales_b, out)) {
    auto out_dtype = (out.dtype().code == kDLBfloat) ? torch::kBFloat16 : torch::kHalf;
    auto ta = fp8abi::view_as(a, torch::kFloat8_e4m3fn);
    auto tb = fp8abi::view_as(b, torch::kFloat8_e4m3fn);
    auto tsa = fp8abi::view_as(scales_a, torch::kFloat32);
    auto tsb = fp8abi::view_as(scales_b, torch::kFloat32);
    auto tout = fp8abi::view_as(out, out_dtype);
    fp8_scaled_mm_baseline_impl(tout, ta, tb, tsa, tsb, c10::nullopt);
    return;
  }
  // (specialized fast path goes here)
}

// Route diagnostic: 1 = candidate specialized fast path, 0 = baseline fallback.
// Launches nothing. Used by bench/correctness.py to PROVE route coverage so a
// silent fallback can never masquerade as a candidate run.
int64_t fp8_scaled_mm_candidate_route(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::TensorView out) {
  return candidate_covers(a, b, scales_a, scales_b, out) ? 1 : 0;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_scaled_mm_candidate, fp8_scaled_mm_candidate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_scaled_mm_candidate_route, fp8_scaled_mm_candidate_route);
