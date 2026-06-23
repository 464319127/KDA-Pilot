// Native-CUDA candidate for `verify_tree_greedy`, specialized for the captured
// GLM-5.2 regime (num_draft_tokens == 2, num_spec_step == 2, small batch).
//
// Design rationale (see ../docs/baseline_source.md and ../docs/results.md):
// this is a microscopic, launch/scheduler-bound kernel — at most bs*nd = 20 int64
// slots of work per call. The upstream baseline launches `batch_size` one-thread
// blocks. The only credible kernel-level lever is launch geometry: this candidate
// runs ONE thread per request packed into a single (or few) block(s), trimming the
// block count from `batch_size` to ~1, with read-only __ldg loads and __restrict__
// pointers. It does not, and cannot, remove the fixed kernel-launch overhead, so a
// no-go is a legitimate possible outcome.
//
// The candidate is exactly equivalent to the baseline algorithm for nd==2,nss==2.
// Any shape/parameter combination outside the specialized regime falls back to the
// recovered baseline so correctness is never lost.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "verify_tree_greedy_kernel.cuh"  // baseline_vtg::launch_baseline (fallback)

namespace candidate_vtg {

// One thread per request. Specialized constants nd==2, nss==2 let the compiler
// drop the speculative-step loop (single accept step) and the multi-child walk
// (a width-2 tree has at most the root's single child plus its sibling chain).
__global__ void VerifyTreeGreedyLaneW2S2(
    int32_t* __restrict__ predicts,
    int32_t* __restrict__ accept_index,
    int32_t* __restrict__ accept_token_num,
    const int64_t* __restrict__ candidates,
    const int64_t* __restrict__ retrive_index,
    const int64_t* __restrict__ retrive_next_token,
    const int64_t* __restrict__ retrive_next_sibling,
    const int64_t* __restrict__ target_predict,
    uint32_t batch_size) {
  const uint32_t bx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bx >= batch_size) return;

  constexpr uint32_t nd = 2;   // num_draft_tokens
  constexpr uint32_t nss = 2;  // num_spec_step

  int64_t last_accepted = __ldg(&retrive_index[bx * nd]);  // global index of root
  accept_index[bx * nss] = static_cast<int32_t>(last_accepted);
  int32_t num_accepted = 0;

  // nss == 2 => exactly one speculative step. Walk the root's child sibling chain.
  int64_t cur = __ldg(&retrive_next_token[bx * nd]);  // first child (row-local slot)
  while (cur != -1) {
    int64_t draft_index = __ldg(&retrive_index[bx * nd + cur]);
    int64_t draft_token = __ldg(&candidates[bx * nd + cur]);
    int64_t target_token = __ldg(&target_predict[last_accepted]);
    if (draft_token == target_token) {
      predicts[last_accepted] = static_cast<int32_t>(target_token);
      num_accepted = 1;
      accept_index[bx * nss + 1] = static_cast<int32_t>(draft_index);
      last_accepted = draft_index;
      break;
    }
    cur = __ldg(&retrive_next_sibling[bx * nd + cur]);
  }

  accept_token_num[bx] = num_accepted;
  predicts[last_accepted] = static_cast<int32_t>(__ldg(&target_predict[last_accepted]));
}

constexpr uint32_t kLaneBlock = 128;  // threads/block (captured production bs<=10 -> 1 block)

// Cheap host-side dispatch (reads tensor metadata only; no device sync on the hot
// path): the captured shape family goes to the specialized candidate, everything else
// falls back to the recovered baseline.
inline void dispatch_verify_tree_greedy(
    int32_t* predicts,
    int32_t* accept_index,
    int32_t* accept_token_num,
    const int64_t* candidates,
    const int64_t* retrive_index,
    const int64_t* retrive_next_token,
    const int64_t* retrive_next_sibling,
    const int64_t* target_predict,
    uint32_t batch_size,
    uint32_t num_spec_step,
    uint32_t num_draft_tokens,
    cudaStream_t stream) {
  // Specialize the captured shape family (num_draft_tokens==2, num_spec_step==2) for any
  // batch size; the lane-per-request kernel scales via grid = ceil(bs / kLaneBlock).
  // Captured production is bs<=10. Every uncovered shape falls back to the recovered baseline.
  const bool specialized = (num_draft_tokens == 2 && num_spec_step == 2);
  if (specialized) {
    const uint32_t block = kLaneBlock;
    const uint32_t grid = (batch_size + block - 1) / block;
    VerifyTreeGreedyLaneW2S2<<<grid, block, 0, stream>>>(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        batch_size);
  } else {
    baseline_vtg::launch_baseline(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        batch_size,
        num_spec_step,
        num_draft_tokens,
        stream);
  }
}

}  // namespace candidate_vtg
