// Candidate entry point for fast_topk_transform_fused (native CUDA).
//
// Workspace-owned candidate. The dominant captured regime is the "decode" naive path
// (length <= topk, no score read): the recovered baseline writes, per token row b with
// seq == b (when row_starts is absent and S == B),
//     dst_page_table[b, i] = (i < lengths[b]) ? src_page_table[b, i] : -1   for i in [0, topk)
// (baseline/sgl-kernel/csrc/elementwise/topk.cu: naive_topk_transform + the decode kernel).
//
// This file provides a native CUDA kernel for exactly that bucket and falls back to the
// recovered SGLang baseline for every other shape / parameter combination, so the candidate
// stays correctness-identical to the baseline everywhere it does not specialize.
//
// Dispatch bucket (decided from tensor metadata only — no host sync, no host read of lengths):
//     topk == 2048 && row_starts == None && S == B && min(N, M) <= 2048
// On that bucket every length <= min(N,M) <= topk, so it is purely the naive path (no score
// read, fully deterministic, exact-match vs baseline). The kernel writes all topk outputs per
// row with one CTA per (row, 256-column tile) and one thread per column: stores to the
// contiguous dst row are fully coalesced, and multiple CTAs per row lift occupancy for the
// small batch sizes typical of decode. Uncovered cases (radix length>topk, ragged row_starts,
// prefill S!=B, min(N,M)>topk, unsupported topk, bad alignment) go to the baseline interface.
//
// Signature mirrors the recovered C++ op exactly (destination-passing dst_page_table). The
// kernel launches on at::cuda::getCurrentCUDAStream() (torch's current stream); device is set
// by the CUDAGuard in binding.cu before this is called.

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>

// Defined in baseline/sgl-kernel/csrc/elementwise/topk.cu (external linkage).
void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts);

namespace {

constexpr int kDecodeTileCols = 256;  // columns per CTA tile (topk==2048 -> 8 tiles per row)

// Exact-naive decode copy/fill for the dominant bucket (seq == b; no score read).
//   dst[b, col] = (col < lengths[b]) ? src_page_table[b, col] : -1
// One thread per output column; blockIdx.x enumerates (row, tile) pairs.
__global__ void decode_copy_fill_kernel(
    const int* __restrict__ src_page_table,  // (S == B, M) int32, column stride 1
    const int* __restrict__ lengths,         // (B,) int32
    int* __restrict__ dst_page_table,        // (B, topk) int32, contiguous
    int batch,
    int topk,
    long src_row_stride,
    long len_stride,
    int tiles_per_row) {
  const int row_tile = blockIdx.x;
  const int b = row_tile / tiles_per_row;
  if (b >= batch) return;
  const int tile = row_tile - b * tiles_per_row;
  const int col = tile * kDecodeTileCols + threadIdx.x;
  if (col >= topk) return;

  const int len_b = lengths[(long)b * len_stride];
  // col < len_b implies col < min(N,M) <= M (lengths are capped to min(N,M) by the contract),
  // so the page-table read stays in bounds; otherwise pad with -1 (no read).
  const int val = (col < len_b) ? src_page_table[(long)b * src_row_stride + col] : -1;
  dst_page_table[(long)b * topk + col] = val;
}

}  // namespace

void fast_topk_transform_candidate(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts) {
  // Cheap rank guards FIRST so the size() reads below cannot throw on an out-of-contract shape
  // (an unexpected rank must fall back to the baseline's own TORCH_CHECKs, not abort here).
  const bool dims_ok =
      dst_page_table.dim() == 2 && score.dim() == 2 && src_page_table.dim() == 2 &&
      lengths.dim() == 1 && cu_seqlens_q.dim() == 1;

  const int64_t batch = dims_ok ? dst_page_table.size(0) : 0;
  const int64_t topk = dims_ok ? dst_page_table.size(1) : 0;
  const int64_t seq_n = dims_ok ? score.size(1) : 0;
  const int64_t num_seq = dims_ok ? src_page_table.size(0) : 0;
  const int64_t page_m = dims_ok ? src_page_table.size(1) : 0;

  // Metadata-only dispatch: no host read of lengths, no sync, no allocation. The predicate is a
  // SUPERSET of the recovered baseline's metadata contract (baseline/.../topk.cu get_params +
  // launcher TORCH_CHECKs), so any public-ABI input the baseline would reject — or that this bucket
  // does not cover — takes fast_topk_transform_interface instead of the native kernel. Every kernel
  // access (lengths[0..B), src[b,0..len_b), dst[b,0..2048)) is provably in-bounds under these guards.
  const bool decode_naive_bucket =
      dims_ok &&
      topk == 2048 &&
      !row_starts.has_value() &&
      num_seq == batch &&                          // decode: one src-page row per token row (seq == b)
      std::min(seq_n, page_m) <= 2048 &&           // length <= min(N,M) <= topk -> naive path
      // score (read only for the metadata above; guarded to match the baseline's score contract)
      score.size(0) == batch &&
      score.scalar_type() == at::kFloat &&
      score.stride(1) == 1 &&
      score.is_cuda() &&
      // lengths (kernel reads lengths[0..B))
      lengths.size(0) == batch &&
      lengths.scalar_type() == at::kInt &&
      lengths.is_contiguous() &&
      lengths.is_cuda() &&
      // cu_seqlens_q (decode contract; kernel does not read it, but the baseline requires it)
      cu_seqlens_q.size(0) == batch + 1 &&
      cu_seqlens_q.is_contiguous() &&
      cu_seqlens_q.is_cuda() &&
      // src_page_table (kernel gathers src[b, col], col-stride 1)
      src_page_table.scalar_type() == at::kInt &&
      src_page_table.stride(1) == 1 &&
      src_page_table.is_cuda() &&
      // dst_page_table (kernel writes contiguous dst[b, 0..2048))
      dst_page_table.scalar_type() == at::kInt &&
      dst_page_table.is_contiguous() &&
      dst_page_table.is_cuda();

  // Optional, one-time-initialized diagnostic (no-op unless TOPK_CANDIDATE_DEBUG is set):
  // lets a correctness/probe run confirm which calls take the native bucket vs the fallback.
  static const bool kDebug = std::getenv("TOPK_CANDIDATE_DEBUG") != nullptr;

  if (!decode_naive_bucket) {
    if (kDebug) {
      std::fprintf(stderr, "[topk-candidate] fallback B=%ld N=%ld M=%ld S=%ld topk=%ld rs=%d dims_ok=%d\n",
                   (long)batch, (long)seq_n, (long)page_m, (long)num_seq, (long)topk,
                   (int)row_starts.has_value(), (int)dims_ok);
    }
    // Every uncovered shape / parameter -> recovered SGLang baseline (identical output).
    fast_topk_transform_interface(score, lengths, dst_page_table, src_page_table, cu_seqlens_q, row_starts);
    return;
  }
  if (kDebug) {
    std::fprintf(stderr, "[topk-candidate] bucket1 B=%ld N=%ld M=%ld\n",
                 (long)batch, (long)seq_n, (long)page_m);
  }

  const int tiles_per_row = static_cast<int>(topk / kDecodeTileCols);  // 2048 / 256 = 8
  const dim3 grid(static_cast<unsigned int>(batch * tiles_per_row));
  const dim3 block(kDecodeTileCols);
  auto stream = at::cuda::getCurrentCUDAStream();
  decode_copy_fill_kernel<<<grid, block, 0, stream>>>(
      src_page_table.data_ptr<int>(),
      lengths.data_ptr<int>(),
      dst_page_table.data_ptr<int>(),
      static_cast<int>(batch),
      static_cast<int>(topk),
      src_page_table.stride(0),
      lengths.stride(0),
      tiles_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();  // non-synchronizing: surfaces a launch-config error immediately
}
