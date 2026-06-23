// Task-local TVM-FFI ABI for fast_topk_transform_fused.
//
// Exposes the recovered baseline and the workspace candidate through the same
// destination-passing signature via tvm::ffi::TensorView. tvm_ffi.cpp.load_inline
// auto-generates the TVM_FFI_DLL_EXPORT_TYPED_FUNC exports for the names listed in its
// `functions=[...]` argument, so the two entry points below are global functions named
// exactly `fast_topk_transform_fused_baseline` / `fast_topk_transform_fused_candidate`
// (no manual export macro, no anonymous namespace for them).
//
// The recovered baseline is ATen-based, so each TensorView is bridged to an at::Tensor
// view via at::from_blob (no copy; preserves strides, incl. the row-strided non-contiguous
// score). Kernels launch on at::cuda::getCurrentCUDAStream() (torch's current stream).
//
// Python-facing signature (dst_out preallocated by the harness, excluded from timing):
//   fast_topk_transform_fused_{baseline,candidate}(
//       score, lengths, page_table_size_1, cu_seqlens_q, topk, row_starts, dst_out)

#include <tvm/ffi/function.h>
#include <tvm/ffi/container/tensor.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <optional>
#include <vector>

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

// Defined earlier in the concatenated translation unit (topk.cu / candidate_topk_transform.cu).
void fast_topk_transform_interface(
    const at::Tensor& score, const at::Tensor& lengths, at::Tensor& dst_page_table,
    const at::Tensor& src_page_table, const at::Tensor& cu_seqlens_q, std::optional<at::Tensor> row_starts);
void fast_topk_transform_candidate(
    const at::Tensor& score, const at::Tensor& lengths, at::Tensor& dst_page_table,
    const at::Tensor& src_page_table, const at::Tensor& cu_seqlens_q, std::optional<at::Tensor> row_starts);

// Zero-copy view of a TensorView as an at::Tensor with the given scalar type.
static at::Tensor as_tensor(const TensorView& tv, at::ScalarType st) {
  DLDevice dev = tv.device();
  auto sh = tv.shape();
  auto stld = tv.strides();
  std::vector<int64_t> sizes(sh.begin(), sh.end());
  std::vector<int64_t> strides(stld.begin(), stld.end());
  auto opts = at::TensorOptions().dtype(st).device(at::kCUDA, static_cast<c10::DeviceIndex>(dev.device_id));
  return at::from_blob(tv.data_ptr(), sizes, strides, opts);
}

// Exported (names MUST match build.py's load_inline functions=[...]; load_inline emits the
// TVM_FFI_DLL_EXPORT_TYPED_FUNC wrappers for these).
void fast_topk_transform_fused_baseline(
    TensorView score, TensorView lengths, TensorView page_table_size_1,
    TensorView cu_seqlens_q, int64_t topk, Optional<TensorView> row_starts, TensorView dst_out) {
  TORCH_CHECK(topk == 2048, "fast_topk_transform_fused is specialized for topk==2048");
  const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(score.device().device_id));
  auto s = as_tensor(score, at::kFloat);
  auto l = as_tensor(lengths, at::kInt);
  auto pts = as_tensor(page_table_size_1, at::kInt);
  auto cu = as_tensor(cu_seqlens_q, at::kInt);
  auto dst = as_tensor(dst_out, at::kInt);
  std::optional<at::Tensor> rs;
  if (row_starts.has_value()) rs = as_tensor(row_starts.value(), at::kInt);
  fast_topk_transform_interface(s, l, dst, pts, cu, rs);
}

void fast_topk_transform_fused_candidate(
    TensorView score, TensorView lengths, TensorView page_table_size_1,
    TensorView cu_seqlens_q, int64_t topk, Optional<TensorView> row_starts, TensorView dst_out) {
  TORCH_CHECK(topk == 2048, "fast_topk_transform_fused is specialized for topk==2048");
  const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(score.device().device_id));
  auto s = as_tensor(score, at::kFloat);
  auto l = as_tensor(lengths, at::kInt);
  auto pts = as_tensor(page_table_size_1, at::kInt);
  auto cu = as_tensor(cu_seqlens_q, at::kInt);
  auto dst = as_tensor(dst_out, at::kInt);
  std::optional<at::Tensor> rs;
  if (row_starts.has_value()) rs = as_tensor(row_starts.value(), at::kInt);
  fast_topk_transform_candidate(s, l, dst, pts, cu, rs);
}
