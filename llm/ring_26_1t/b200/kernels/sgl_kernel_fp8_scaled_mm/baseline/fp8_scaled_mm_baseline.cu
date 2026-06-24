// Destination-passing ABI wrapper for the recovered SGLang fp8_scaled_mm
// baseline. This TU is the ONLY one that compiles the verbatim recovered
// fp8_gemm_kernel.cu (it brings in sm100_fp8_dispatch_shape<>, getSMVersion(),
// etc.). The candidate TU calls fp8_scaled_mm_baseline_impl() (declared in
// bench/csrc/fp8_scaled_mm_abi.h) for its fallback, so the upstream source is
// compiled exactly once and both sides share one build.
//
// Include order matters: torch/cuda headers first, then the verbatim baseline,
// then the shared ABI header.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

// Verbatim recovered upstream kernel (sm89/sm90/sm100/sm120 dispatch). Compiled
// here only. Resolves "math.hpp", "utils.h", "cutlass_extensions/..." with
// -I <baseline dir>; CUTLASS/CuTe headers with the pinned CUTLASS include dirs.
#include "fp8_gemm_kernel.cu"

#include "fp8_scaled_mm_abi.h"

// Destination-passing baseline: write the result into the pre-allocated `out`.
// Mirrors the upstream entry's arch/out-dtype dispatch but skips the internal
// torch::empty allocation (the harness pre-allocates `out`), so timing is fair.
void fp8_scaled_mm_baseline_impl(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  // Mirror the upstream fp8_scaled_mm entry's input contract (the entry's
  // TORCH_CHECKs are bypassed because we call the dispatch directly for
  // destination passing). Faithful + fair: both sides pay these cheap checks,
  // and invalid inputs (e.g. a contiguous/row-major B) are rejected exactly as
  // the upstream public op rejects them, instead of silently mis-computing.
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a, b must be CUDA tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a, b must be 2D");
  TORCH_CHECK(a.stride(1) == 1, "mat_a must be row major");
  TORCH_CHECK(b.stride(0) == 1, "mat_b must be column major");
  TORCH_CHECK(a.size(1) == b.size(0), "a, b shapes cannot be multiplied");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "mat_a must be Float8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "mat_b must be Float8_e4m3fn");
  TORCH_CHECK(out.scalar_type() == torch::kHalf || out.scalar_type() == torch::kBFloat16,
              "out_dtype must be Half or BFloat16");
  TORCH_CHECK(scales_a.numel() == a.size(0) && scales_b.numel() == b.size(1), "scale size mismatch");
  TORCH_CHECK(scales_a.is_contiguous() && scales_b.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32 && scales_b.scalar_type() == torch::kFloat32,
              "scales must be Float32");
  auto sm_version = getSMVersion();
#if defined CUDA_VERSION && CUDA_VERSION >= 12080
  if (sm_version >= 120) {
    if (out.scalar_type() == torch::kBFloat16)
      sm120_fp8_dispatch_shape<cutlass::bfloat16_t>(out, a, b, scales_a, scales_b, bias);
    else
      sm120_fp8_dispatch_shape<cutlass::half_t>(out, a, b, scales_a, scales_b, bias);
    return;
  } else if (sm_version >= 100) {
    if (out.scalar_type() == torch::kBFloat16)
      sm100_fp8_dispatch_shape<cutlass::bfloat16_t>(out, a, b, scales_a, scales_b, bias);
    else
      sm100_fp8_dispatch_shape<cutlass::half_t>(out, a, b, scales_a, scales_b, bias);
    return;
  }
#endif
#if defined CUDA_VERSION && CUDA_VERSION >= 12000
  if (sm_version >= 90) {
    cutlass_scaled_mm_sm90_fp8(out, a, b, scales_a, scales_b, bias);
    return;
  }
#endif
#if defined CUDA_VERSION && CUDA_VERSION >= 12040
  if (sm_version == 89) {
    if (out.scalar_type() == torch::kBFloat16)
      sm89_fp8_dispatch_shape<cutlass::bfloat16_t>(out, a, b, scales_a, scales_b, bias);
    else
      sm89_fp8_dispatch_shape<cutlass::half_t>(out, a, b, scales_a, scales_b, bias);
    return;
  }
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No fp8_scaled_mm baseline for sm", sm_version);
}

// TVM-FFI export: TensorView (a, b, scale_a, scale_b, out) -> non-owning torch
// views -> destination-passing impl. Inputs known-dtype (fp8e4m3 A/B, fp32
// scales); out dtype taken from the pre-allocated output's DLDataType.
void fp8_scaled_mm_baseline(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::TensorView out) {
  // Reject contract-violating dtypes at the TensorView boundary before the
  // forced-dtype view (which would otherwise reinterpret e.g. e5m2/uint8 as e4m3fn).
  fp8abi::require_fp8_contract(a, b, scales_a, scales_b, out);
  auto out_dtype = (out.dtype().code == kDLBfloat) ? torch::kBFloat16 : torch::kHalf;
  auto ta = fp8abi::view_as(a, torch::kFloat8_e4m3fn);
  auto tb = fp8abi::view_as(b, torch::kFloat8_e4m3fn);
  auto tsa = fp8abi::view_as(scales_a, torch::kFloat32);
  auto tsb = fp8abi::view_as(scales_b, torch::kFloat32);
  auto tout = fp8abi::view_as(out, out_dtype);
  fp8_scaled_mm_baseline_impl(tout, ta, tb, tsa, tsb, c10::nullopt);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_scaled_mm_baseline, fp8_scaled_mm_baseline);
