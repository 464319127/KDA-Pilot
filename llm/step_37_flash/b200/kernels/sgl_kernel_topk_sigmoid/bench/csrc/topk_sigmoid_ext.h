// Shared declarations + helpers for the standalone topk_sigmoid baseline + candidate,
// exposed through the repo's local direct-symbol ABI: TVM-FFI
// (`TVM_FFI_DLL_EXPORT_TYPED_FUNC` / `tvm::ffi::TensorView`), destination passing
// (topk_weights/topk_indices pre-allocated and written in place; gating_output read-only;
// returns void/None), every launch on at::cuda::getCurrentCUDAStream(). Baseline (the
// recovered upstream kernel, baseline/topk_sigmoid_baseline.cu) and the native-CUDA
// candidate (solution/topk_sigmoid_candidate.cuh) are compiled together in one module with
// identical flags so both sides share the same registration/export/build style and call
// path (the fairness requirement).
#pragma once

#include <cstdint>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace tse {
using tvm::ffi::TensorView;

// --- dtype helpers (DLDataType) ---
inline bool dtype_is(DLDataType d, uint8_t code, uint8_t bits) {
  return d.code == code && d.bits == bits && d.lanes == 1;
}
inline bool is_f32(DLDataType d) { return dtype_is(d, kDLFloat, 32); }
inline bool is_i32(DLDataType d) { return dtype_is(d, kDLInt, 32); }

// --- view helpers ---
template <typename T>
inline T* mptr(const TensorView& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data_ptr()) + t.byte_offset());
}
template <typename T>
inline const T* dptr(const TensorView& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data_ptr()) + t.byte_offset());
}
inline bool is_contiguous(const TensorView& t) {
  for (int i = 0; i < t.ndim(); ++i) {
    if (t.size(i) == 0) return true;  // zero-element tensors are contiguous by convention
  }
  int64_t expect = 1;
  for (int i = t.ndim() - 1; i >= 0; --i) {
    if (t.size(i) == 1) continue;  // stride-free on size-1 dims
    if (t.stride(i) != expect) return false;
    expect *= t.size(i);
  }
  return true;
}
inline int64_t numel(const TensorView& t) {
  int64_t n = 1;
  for (int i = 0; i < t.ndim(); ++i) n *= t.size(i);
  return n;
}
inline bool is_cuda(const TensorView& t) { return t.device().device_type == kDLCUDA; }
inline int device_id(const TensorView& t) { return t.device().device_id; }
}  // namespace tse

// Recovered upstream baseline, exposed through the local TVM-FFI ABI (bridges TensorView ->
// torch::Tensor and calls the vendored topk_sigmoid(...); device kernels are verbatim, see
// baseline/topk_sigmoid_baseline.cu). renormalize is passed as int64 (0/1) for ABI uniformity.
void topk_sigmoid_baseline(
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView topk_indices,
    tvm::ffi::TensorView gating_output,
    int64_t renormalize,
    tvm::ffi::TensorView correction_bias);

// Native-CUDA candidate. Single fused launch on the validated contract (fp32 [N,288],
// topk=8, num_experts=288, renormalize, bias present, contiguous); any other combination
// falls back to topk_sigmoid_baseline so correctness is never lost.
void topk_sigmoid_candidate(
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView topk_indices,
    tvm::ffi::TensorView gating_output,
    int64_t renormalize,
    tvm::ffi::TensorView correction_bias);

// Empty-kernel launch-floor probe (same grid/block shape the candidate fast path uses).
void topk_sigmoid_noop(tvm::ffi::TensorView gating_output);

// Dispatch-route diagnostic: 1 if these inputs take the candidate native fast path, 0 if they
// fall back to the baseline. Runs the SAME predicate as topk_sigmoid_candidate, launches
// nothing — used by bench/correctness.py to PROVE route coverage so a silent fallback can
// never masquerade as a candidate run.
int64_t topk_sigmoid_candidate_route(
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView topk_indices,
    tvm::ffi::TensorView gating_output,
    int64_t renormalize,
    tvm::ffi::TensorView correction_bias);
