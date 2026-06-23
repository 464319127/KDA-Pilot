// Shared declarations for the standalone build_tree_kernel_efficient baseline +
// candidate. Both sides expose the identical destination-passing ABI:
// outputs are pre-allocated and passed last-but-for-scalars; the function returns
// void (Python None) and mutates the output tensors in place; every launch uses
// at::cuda::getCurrentCUDAStream(). This mirrors the upstream
// sgl_kernel.build_tree_kernel_efficient op signature exactly.
#pragma once
#include <ATen/ATen.h>

// Recovered upstream baseline (verbatim kernel bodies; see baseline/).
void build_tree_baseline(
    at::Tensor parent_list,
    at::Tensor selected_index,
    at::Tensor verified_seq_len,
    at::Tensor tree_mask,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode);

// Native-CUDA candidate (specialized fast path for the captured GLM-5.2 regime
// topk=1, depth=1, draft_token_num=2, FULL_MASK, contiguous int64/bool; every
// other shape/dtype/scalar combination falls back to build_tree_baseline).
void build_tree_candidate(
    at::Tensor parent_list,
    at::Tensor selected_index,
    at::Tensor verified_seq_len,
    at::Tensor tree_mask,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode);

// Empty-kernel launch-floor probe: launches a do-nothing kernel with the same
// grid/block the candidate fast path uses (one block, bs threads). Used to
// measure the irreducible launch/scheduling latency on the target GPU.
void build_tree_noop(at::Tensor verified_seq_len, int64_t draft_token_num);
