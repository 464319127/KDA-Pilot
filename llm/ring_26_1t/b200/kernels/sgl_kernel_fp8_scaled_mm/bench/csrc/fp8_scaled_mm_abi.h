// Shared declarations + helpers for the standalone sgl_kernel.fp8_scaled_mm
// baseline + candidate, exposed through the repo's local direct-symbol ABI:
// TVM-FFI (`TVM_FFI_DLL_EXPORT_TYPED_FUNC` / `tvm::ffi::TensorView`),
// DESTINATION-PASSING (the output tensor is pre-allocated by the harness and
// passed LAST, mutated in place; the exported functions return void), every
// CUDA launch on at::cuda::getCurrentCUDAStream(). Baseline and candidate are
// compiled together in ONE module with identical flags so both sides share the
// exact same registration/export/build style and call path (the fairness
// requirement in llm/docs/standalone_llm_benchmark.md).
//
// The upstream Python op `sgl_kernel.fp8_scaled_mm(a, b, scale_a, scale_b,
// out_dtype, bias)` is return-value style (it allocates `out` internally). Here
// the harness pre-allocates `out` and passes it last; the baseline impl calls
// the upstream sm100 dispatch (which already takes `out` by reference), so no
// extra allocation or copy is timed on either side.
#pragma once

#include <cstdint>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <dlpack/dlpack.h>
#include <torch/all.h>
#include <tvm/ffi/container/tensor.h>

namespace fp8abi {
using tvm::ffi::TensorView;

// Build a NON-OWNING torch view over the memory a TensorView points at, with an
// explicit torch dtype (the captured kernel dtypes are fixed and known, so we do
// not need a DLDataType->torch mapping table on the hot path). Strides are taken
// verbatim from the TensorView, so the captured column-major B (stride (1,K)) is
// preserved and the upstream `b.stride(0)==1` check passes.
inline torch::Tensor view_as(const TensorView& t, torch::ScalarType dtype) {
  std::vector<int64_t> sizes, strides;
  sizes.reserve(t.ndim());
  strides.reserve(t.ndim());
  for (int i = 0; i < t.ndim(); ++i) {
    sizes.push_back(t.size(i));
    strides.push_back(t.stride(i));
  }
  void* ptr = static_cast<char*>(t.data_ptr()) + t.byte_offset();
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, t.device().device_id);
  return torch::from_blob(ptr, sizes, strides, opts);
}

inline bool is_cuda(const TensorView& t) { return t.device().device_type == kDLCUDA; }

inline bool dtype_is(const TensorView& t, DLDataTypeCode code, uint8_t bits) {
  return t.dtype().code == code && t.dtype().bits == bits && t.dtype().lanes == 1;
}

// Validate the captured input contract at the TensorView boundary BEFORE building
// any forced-dtype torch view. `view_as` reinterprets bytes at a fixed torch
// dtype, so without this guard a wrong-dtype input (e.g. float8_e5m2 / uint8 A)
// would be silently reinterpreted as float8_e4m3fn and skip the impl's dtype
// check. Both the baseline export and the candidate fallback call this so invalid
// inputs are rejected (not mis-computed). Mirrors the upstream op's dtype checks.
inline void require_fp8_contract(
    const TensorView& a, const TensorView& b,
    const TensorView& sa, const TensorView& sb, const TensorView& out) {
  TORCH_CHECK(dtype_is(a, kDLFloat8_e4m3fn, 8), "fp8_scaled_mm: mat_a must be float8_e4m3fn");
  TORCH_CHECK(dtype_is(b, kDLFloat8_e4m3fn, 8), "fp8_scaled_mm: mat_b must be float8_e4m3fn");
  TORCH_CHECK(dtype_is(sa, kDLFloat, 32) && dtype_is(sb, kDLFloat, 32), "fp8_scaled_mm: scales must be float32");
  TORCH_CHECK(out.dtype().code == kDLBfloat || (out.dtype().code == kDLFloat && out.dtype().bits == 16),
              "fp8_scaled_mm: out_dtype must be bfloat16 or float16");
}
}  // namespace fp8abi

// Recovered upstream baseline, destination-passing (writes into pre-allocated
// `out`). Defined in baseline/fp8_scaled_mm_baseline.cu, which compiles the
// verbatim recovered fp8_gemm_kernel.cu in that single TU. The candidate TU
// calls this for its fallback, so the upstream source is compiled exactly once.
void fp8_scaled_mm_baseline_impl(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias);
