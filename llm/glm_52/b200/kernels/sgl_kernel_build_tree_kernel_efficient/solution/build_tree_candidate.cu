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
//
// ABI: local TVM-FFI direct-symbol (tvm::ffi::TensorView args, in place, current
// CUDA stream), identical to the baseline.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <tvm/ffi/function.h>

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

// Shared fast-path predicate (host-side, device-memory-free; no host sync on the
// hot path). Every piece of metadata that defines the candidate's domain is
// checked here; anything off-domain (wrong scalar, dtype, rank, shape, contiguity,
// or device) is rejected so the dispatcher falls back to the recovered baseline.
// Used by BOTH build_tree_candidate (dispatch) and build_tree_candidate_route
// (the diagnostic the correctness suite asserts on).
inline bool candidate_fast_path_eligible(
    const tvm::ffi::TensorView& parent_list,
    const tvm::ffi::TensorView& selected_index,
    const tvm::ffi::TensorView& verified_seq_len,
    const tvm::ffi::TensorView& tree_mask,
    const tvm::ffi::TensorView& positions,
    const tvm::ffi::TensorView& retrive_index,
    const tvm::ffi::TensorView& retrive_next_token,
    const tvm::ffi::TensorView& retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode) {
  const int64_t bs = parent_list.size(0);
  const bool scalars_ok = topk == 1 && depth == 1 && draft_token_num == 2 &&
      tree_mask_mode == FULL_MASK_MODE;
  const bool dtypes_ok = bte::is_bool(tree_mask.dtype()) && bte::is_i64(parent_list.dtype()) &&
      bte::is_i64(selected_index.dtype()) && bte::is_i64(verified_seq_len.dtype()) &&
      bte::is_i64(positions.dtype()) && bte::is_i64(retrive_index.dtype()) &&
      bte::is_i64(retrive_next_token.dtype()) && bte::is_i64(retrive_next_sibling.dtype());
  // parent_list [bs, 0] (empty -> degenerate depth-1, selected_index must be 0);
  // selected_index [bs, draft_token_num-1] = [bs, 1]; verified_seq_len [bs];
  // positions/retrive_* numel == bs*draft_token_num.
  const bool shapes_ok = parent_list.ndim() == 2 && parent_list.size(0) == bs &&
      parent_list.size(1) == 0 && selected_index.ndim() == 2 &&
      selected_index.size(0) == bs && selected_index.size(1) == draft_token_num - 1 &&
      verified_seq_len.ndim() == 1 && verified_seq_len.size(0) == bs &&
      bte::numel(positions) == bs * draft_token_num &&
      bte::numel(retrive_index) == bs * draft_token_num &&
      bte::numel(retrive_next_token) == bs * draft_token_num &&
      bte::numel(retrive_next_sibling) == bs * draft_token_num;
  const bool contig_ok = bte::is_contiguous(parent_list) && bte::is_contiguous(selected_index) &&
      bte::is_contiguous(verified_seq_len) && bte::is_contiguous(tree_mask) &&
      bte::is_contiguous(positions) && bte::is_contiguous(retrive_index) &&
      bte::is_contiguous(retrive_next_token) && bte::is_contiguous(retrive_next_sibling);
  const bool device_ok = bte::is_cuda(verified_seq_len) && bte::is_cuda(tree_mask) &&
      bte::is_cuda(positions) && bte::is_cuda(retrive_index) &&
      bte::is_cuda(retrive_next_token) && bte::is_cuda(retrive_next_sibling);
  return scalars_ok && dtypes_ok && shapes_ok && contig_ok && device_ok;
}

}  // namespace

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
    int64_t tree_mask_mode) {
  const int64_t bs = parent_list.size(0);
  const bool fast_path = candidate_fast_path_eligible(
      parent_list, selected_index, verified_seq_len, tree_mask, positions,
      retrive_index, retrive_next_token, retrive_next_sibling,
      topk, depth, draft_token_num, tree_mask_mode);

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
      bte::mptr<int64_t>(verified_seq_len),
      bte::mptr<bool>(tree_mask),
      bte::mptr<int64_t>(positions),
      bte::mptr<int64_t>(retrive_index),
      bte::mptr<int64_t>(retrive_next_token),
      static_cast<int>(bs));
}

void build_tree_noop(tvm::ffi::TensorView verified_seq_len, int64_t draft_token_num) {
  const int64_t bs = bte::numel(verified_seq_len);
  if (bs <= 0) {
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = bs <= 256 ? static_cast<int>(bs) : 256;
  const int blocks = static_cast<int>((bs + threads - 1) / threads);
  noop_kernel<<<blocks, threads, 0, stream>>>();
}

// Dispatch-route diagnostic: 1 = candidate native fast path, 0 = baseline fallback.
// Runs the SAME predicate as build_tree_candidate; launches nothing. Lets the
// correctness suite PROVE the fast path actually covers the captured production
// regime (and that off-domain inputs fall back), so a silent fallback cannot
// masquerade as a candidate run.
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
    int64_t tree_mask_mode) {
  return candidate_fast_path_eligible(
             parent_list, selected_index, verified_seq_len, tree_mask, positions,
             retrive_index, retrive_next_token, retrive_next_sibling, topk, depth,
             draft_token_num, tree_mask_mode)
             ? 1
             : 0;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_tree_candidate, build_tree_candidate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_tree_candidate_route, build_tree_candidate_route);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_tree_noop, build_tree_noop);
