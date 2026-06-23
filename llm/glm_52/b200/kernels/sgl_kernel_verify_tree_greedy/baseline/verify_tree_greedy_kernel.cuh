// Buildable baseline kernel for the standalone harness.
//
// The kernel body below is a VERBATIM copy of upstream `VerifyTreeGreedy` from
// SGLang main @7e6587c94a1d0305815a14067c5d3cc02a9b0f36
// (sgl-kernel/csrc/speculative/eagle_utils.cu, lines 272-313), kept byte-for-byte
// so it serves as the baseline oracle. The full upstream file is preserved
// verbatim alongside this header in `baseline/eagle_utils.cu` for provenance.
// `launch_baseline` reproduces the upstream host launch exactly:
// `dim3 grid(batch_size); dim3 block(1);` on the current CUDA stream.
//
// Do NOT optimize anything in this file. The candidate lives in `solution/`.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace baseline_vtg {

template <typename IdType, typename IdType2>
__global__ void VerifyTreeGreedy(
    IdType* predicts,
    IdType* accept_index,
    IdType* accept_token_num,  // mutable
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    IdType2* target_predict,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens) {
  uint32_t bx = blockIdx.x;

  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      IdType2 target_token_id = target_predict[last_accepted_retrive_idx];

      if (draft_token_id == target_token_id) {
        // accept token
        predicts[last_accepted_retrive_idx] = target_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;
  predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx];
}

// Host launcher reproducing the upstream configuration exactly (grid(bs), block(1)).
inline void launch_baseline(
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
  dim3 grid(batch_size);
  dim3 block(1);
  VerifyTreeGreedy<int32_t, int64_t><<<grid, block, 0, stream>>>(
      predicts,
      accept_index,
      accept_token_num,
      const_cast<int64_t*>(candidates),
      const_cast<int64_t*>(retrive_index),
      const_cast<int64_t*>(retrive_next_token),
      const_cast<int64_t*>(retrive_next_sibling),
      const_cast<int64_t*>(target_predict),
      batch_size,
      num_spec_step,
      num_draft_tokens);
}

}  // namespace baseline_vtg
