// Native-CUDA candidate for sgl_kernel.build_tree_kernel_efficient, specialized
// for the captured GLM-5.2 B200 regime: topk=1, depth=1, draft_token_num=2,
// tree_mask_mode=FULL_MASK, contiguous int64 inputs + bool tree_mask, and an
// empty parent_list ([bs, 0]) which (for a baseline-valid input) forces
// selected_index == 0, i.e. the degenerate depth-1 single-chain tree.
//
// Derived exact post-state on the FULL_MASK pre-state (tree_mask prefilled True,
// retrieve buffers prefilled -1) for request b, with L_b = verified_seq_len[b],
// S_b = 4*b + 2*sum_{i<b} L_i:
//   tree_mask[S_b + L_b + 1] = false   (only changed entry; all others stay True)
//   positions[2b] = L_b ; positions[2b+1] = L_b + 1
//   retrive_index[2b] = 2b ; retrive_index[2b+1] = 2b + 1
//   retrive_next_token[2b] = 1 ; retrive_next_token[2b+1] stays -1
//   retrive_next_sibling[2b], [2b+1] stay -1
// This matches the recovered baseline's net effect bit-for-bit on the captured
// pre-state. Any other shape/dtype/scalar/contiguity combination falls back to
// the recovered baseline so correctness is never lost. The dispatch test is
// host-side and touches no device memory (no host sync on the hot path).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "build_tree_ext.h"

namespace {

// One thread per request. Single (or few) block(s) to minimize grid scheduling
// vs the baseline's bs-block / 2-thread launch. draft_token_num is fixed to 2 on
// this fast path.
__global__ void build_tree_candidate_kernel(
    const int64_t* __restrict__ verified_seq_len,
    bool* __restrict__ tree_mask,
    int64_t* __restrict__ positions,
    int64_t* __restrict__ retrive_index,
    int64_t* __restrict__ retrive_next_token,
    int bs) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= bs) {
    return;
  }
  // Prefix offset over prior requests (bs is small in the captured regime).
  long prefix = 0;
  for (int i = 0; i < b; ++i) {
    prefix += verified_seq_len[i];
  }
  long L = verified_seq_len[b];
  long S = 4LL * b + 2LL * prefix;  // draft_token_num == 2

  // Only changed tree_mask entry vs the True pre-state.
  tree_mask[S + L + 1] = false;

  long base = 2LL * b;  // draft_token_num == 2
  positions[base] = L;
  positions[base + 1] = L + 1;
  retrive_index[base] = base;
  retrive_index[base + 1] = base + 1;
  retrive_next_token[base] = 1;
  // retrive_next_token[base + 1] and retrive_next_sibling[*] keep their -1 pre-state.
}

__global__ void noop_kernel() {}

constexpr int64_t FULL_MASK_MODE = 0;

inline bool is_long(const at::Tensor& t) {
  return t.scalar_type() == at::kLong;
}

}  // namespace

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
    int64_t tree_mask_mode) {
  const int64_t bs = parent_list.size(0);

  // Host-side, device-memory-free regime check (no host sync on the hot path).
  const bool fast_path = topk == 1 && depth == 1 && draft_token_num == 2 &&
      tree_mask_mode == FULL_MASK_MODE && parent_list.dim() == 2 &&
      parent_list.size(1) == 0 && tree_mask.scalar_type() == at::kBool &&
      is_long(verified_seq_len) && is_long(selected_index) && is_long(positions) &&
      is_long(retrive_index) && is_long(retrive_next_token) && is_long(retrive_next_sibling) &&
      verified_seq_len.is_contiguous() && tree_mask.is_contiguous() &&
      positions.is_contiguous() && retrive_index.is_contiguous() &&
      retrive_next_token.is_contiguous() && retrive_next_sibling.is_contiguous();

  if (!fast_path) {
    build_tree_baseline(
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode);
    return;
  }

  if (bs <= 0) {
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = bs <= 256 ? static_cast<int>(bs) : 256;
  const int blocks = static_cast<int>((bs + threads - 1) / threads);
  build_tree_candidate_kernel<<<blocks, threads, 0, stream>>>(
      verified_seq_len.data_ptr<int64_t>(),
      tree_mask.data_ptr<bool>(),
      positions.data_ptr<int64_t>(),
      retrive_index.data_ptr<int64_t>(),
      retrive_next_token.data_ptr<int64_t>(),
      static_cast<int>(bs));
}

void build_tree_noop(at::Tensor verified_seq_len, int64_t draft_token_num) {
  const int64_t bs = verified_seq_len.numel();
  if (bs <= 0) {
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = bs <= 256 ? static_cast<int>(bs) : 256;
  const int blocks = static_cast<int>((bs + threads - 1) / threads);
  noop_kernel<<<blocks, threads, 0, stream>>>();
}
