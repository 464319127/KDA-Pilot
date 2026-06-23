// Task-local ABI for fast_topk_transform_fused: exposes the recovered baseline and the
// workspace candidate through the SAME destination-passing signature, so the benchmark
// times both sides through an identical path (no asymmetric wrapper overhead).
//
// Python-facing signature (matches the captured kwargs; dst_out is preallocated by the
// harness and excluded from timing):
//   fast_topk_transform_fused_{baseline,candidate}(
//       score, lengths, page_table_size_1, cu_seqlens_q, topk, row_starts, dst_out)
// maps to the C++ op fast_topk_transform_interface(score, lengths, dst_page_table=dst_out,
//   src_page_table=page_table_size_1, cu_seqlens_q, row_starts).

#include <torch/extension.h>
#include <optional>

void fast_topk_transform_interface(
    const at::Tensor& score, const at::Tensor& lengths, at::Tensor& dst_page_table,
    const at::Tensor& src_page_table, const at::Tensor& cu_seqlens_q, std::optional<at::Tensor> row_starts);

void fast_topk_transform_candidate(
    const at::Tensor& score, const at::Tensor& lengths, at::Tensor& dst_page_table,
    const at::Tensor& src_page_table, const at::Tensor& cu_seqlens_q, std::optional<at::Tensor> row_starts);

namespace {

void baseline_fn(
    const at::Tensor& score, const at::Tensor& lengths, const at::Tensor& page_table_size_1,
    const at::Tensor& cu_seqlens_q, int64_t topk, std::optional<at::Tensor> row_starts, at::Tensor dst_out) {
  TORCH_CHECK(topk == 2048, "fast_topk_transform_fused is specialized for topk==2048");
  fast_topk_transform_interface(score, lengths, dst_out, page_table_size_1, cu_seqlens_q, row_starts);
}

void candidate_fn(
    const at::Tensor& score, const at::Tensor& lengths, const at::Tensor& page_table_size_1,
    const at::Tensor& cu_seqlens_q, int64_t topk, std::optional<at::Tensor> row_starts, at::Tensor dst_out) {
  TORCH_CHECK(topk == 2048, "fast_topk_transform_fused is specialized for topk==2048");
  fast_topk_transform_candidate(score, lengths, dst_out, page_table_size_1, cu_seqlens_q, row_starts);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_topk_transform_fused_baseline", &baseline_fn,
        "Recovered SGLang baseline (destination-passing).",
        py::arg("score"), py::arg("lengths"), py::arg("page_table_size_1"),
        py::arg("cu_seqlens_q"), py::arg("topk"), py::arg("row_starts"), py::arg("dst_out"));
  m.def("fast_topk_transform_fused_candidate", &candidate_fn,
        "Workspace candidate (destination-passing).",
        py::arg("score"), py::arg("lengths"), py::arg("page_table_size_1"),
        py::arg("cu_seqlens_q"), py::arg("topk"), py::arg("row_starts"), py::arg("dst_out"));
}
