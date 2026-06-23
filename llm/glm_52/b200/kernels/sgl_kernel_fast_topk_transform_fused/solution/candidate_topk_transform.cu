// Candidate entry point for fast_topk_transform_fused.
//
// This is the workspace-owned native-CUDA candidate. It currently forwards to the
// recovered baseline so the ABI, harness, and correctness gate can be exercised
// end-to-end and the candidate is correctness-identical to the baseline by construction.
// The native CUDA fast path (N<=topk copy/pad/transform first, then specialized
// large-prefill / small-decode paths + cheap shape dispatch with baseline fallback)
// replaces this forwarding body later, once baseline numbers are frozen.
//
// Signature mirrors the recovered C++ op exactly (destination-passing dst_page_table,
// launches on at::cuda::getCurrentCUDAStream() inside the baseline interface).

#include <ATen/core/Tensor.h>
#include <optional>

// Defined in baseline/sgl-kernel/csrc/elementwise/topk.cu (external linkage).
void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts);

void fast_topk_transform_candidate(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts) {
  // Fallback: identical to the recovered baseline until the native candidate lands.
  fast_topk_transform_interface(score, lengths, dst_page_table, src_page_table, cu_seqlens_q, row_starts);
}
