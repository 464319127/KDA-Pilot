// Shared declarations + helpers for the standalone build_tree_kernel_efficient
// baseline + candidate, exposed through the repo's local direct-symbol ABI:
// TVM-FFI (`TVM_FFI_DLL_EXPORT_TYPED_FUNC` / `tvm::ffi::TensorView`), destination
// passing (outputs pre-allocated and mutated in place; returns void/None), every
// launch on at::cuda::getCurrentCUDAStream(). Mirrors the diffusion/kernels/*
// pattern. Baseline and candidate are compiled together in one module with
// identical flags so both sides share the exact same registration/export/build
// style and call path (the fairness requirement).
#pragma once

#include <cstdint>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace bte {
using tvm::ffi::TensorView;

// --- dtype helpers (DLDataType) ---
inline bool dtype_is(DLDataType d, uint8_t code, uint8_t bits) {
  return d.code == code && d.bits == bits && d.lanes == 1;
}
inline bool is_i64(DLDataType d) { return dtype_is(d, kDLInt, 64); }
inline bool is_bool(DLDataType d) { return dtype_is(d, kDLBool, 8); }

// --- view helpers ---
template <typename T>
inline T* mptr(const TensorView& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data_ptr()) + t.byte_offset());
}
inline bool is_contiguous(const TensorView& t) {
  // A zero-element tensor (any size-0 dim, e.g. the captured parent_list [bs,0])
  // has no elements to stride over and is contiguous by PyTorch convention. This
  // guard is required: without it the compact-stride walk below multiplies the
  // expected stride by the zero dim and then wrongly rejects the leading dim for
  // bs>1, which would silently route the candidate fast path to the baseline.
  for (int i = 0; i < t.ndim(); ++i) {
    if (t.size(i) == 0) return true;
  }
  int64_t expect = 1;
  for (int i = t.ndim() - 1; i >= 0; --i) {
    if (t.size(i) == 1) continue;  // stride free on size-1 dims
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
}  // namespace bte

// Recovered upstream baseline (verbatim device kernels; see baseline/).
void build_tree_baseline(
    tvm::ffi::TensorView parent_list,
    tvm::ffi::TensorView selected_index,
    tvm::ffi::TensorView verified_seq_len,
    tvm::ffi::TensorView tree_mask,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView retrive_index,
    tvm::ffi::TensorView retrive_next_token,
    tvm::ffi::TensorView retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode);

// Native-CUDA candidate (specialized fast path for the captured GLM-5.2 regime;
// any other shape/dtype/scalar/contiguity combination falls back to the baseline).
void build_tree_candidate(
    tvm::ffi::TensorView parent_list,
    tvm::ffi::TensorView selected_index,
    tvm::ffi::TensorView verified_seq_len,
    tvm::ffi::TensorView tree_mask,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView retrive_index,
    tvm::ffi::TensorView retrive_next_token,
    tvm::ffi::TensorView retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode);

// Empty-kernel launch-floor probe (same grid/block the candidate fast path uses).
void build_tree_noop(tvm::ffi::TensorView verified_seq_len, int64_t draft_token_num);

// Dispatch-route diagnostic: returns 1 if these inputs take the candidate native
// fast path, 0 if they fall back to the baseline. Runs the SAME predicate as
// build_tree_candidate, launches nothing — used by bench/correctness.py to PROVE
// route coverage (fast path for the captured production regime; fallback for
// off-domain inputs), so a silent fallback can never masquerade as a candidate run.
int64_t build_tree_candidate_route(
    tvm::ffi::TensorView parent_list,
    tvm::ffi::TensorView selected_index,
    tvm::ffi::TensorView verified_seq_len,
    tvm::ffi::TensorView tree_mask,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView retrive_index,
    tvm::ffi::TensorView retrive_next_token,
    tvm::ffi::TensorView retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode);
