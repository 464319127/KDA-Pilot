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
