// Shared low-overhead local ABI for the standalone baseline-vs-candidate harness.
//
// ONE torch CUDA extension exposes BOTH entry points with the IDENTICAL signature
// (the five int64 inputs first, the three mutable int32 outputs LAST), each launched
// on `at::cuda::getCurrentCUDAStream()` with the same validation. Both sides use the
// same builder path and compile flags (see ../docs/benchmark_method.md), so the
// CUDA-event-timed comparison is fair. Upstream's public signature lists the three
// outputs FIRST; the outputs-first -> outputs-last remapping is documented in
// docs/benchmark_method.md.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>

#include "verify_tree_greedy_kernel.cuh"     // baseline_vtg::launch_baseline   (-I baseline)
#include "verify_tree_greedy_candidate.cuh"  // candidate_vtg::dispatch_...     (-I solution)

namespace {

inline void check_input(const at::Tensor& t, c10::ScalarType dtype, int64_t dim, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has wrong dtype");
  TORCH_CHECK(t.dim() == dim, name, " has wrong rank");
}

// Validate the shared ABI shape/dtype contract for both sides identically.
void validate(
    const at::Tensor& candidates,
    const at::Tensor& retrive_index,
    const at::Tensor& retrive_next_token,
    const at::Tensor& retrive_next_sibling,
    const at::Tensor& target_predict,
    const at::Tensor& predicts,
    const at::Tensor& accept_index,
    const at::Tensor& accept_token_num) {
  check_input(candidates, at::kLong, 2, "candidates");
  check_input(retrive_index, at::kLong, 2, "retrive_index");
  check_input(retrive_next_token, at::kLong, 2, "retrive_next_token");
  check_input(retrive_next_sibling, at::kLong, 2, "retrive_next_sibling");
  check_input(target_predict, at::kLong, 2, "target_predict");
  check_input(predicts, at::kInt, 1, "predicts");
  check_input(accept_index, at::kInt, 2, "accept_index");
  check_input(accept_token_num, at::kInt, 1, "accept_token_num");
  const auto bs = candidates.size(0);
  const auto nd = candidates.size(1);
  TORCH_CHECK(retrive_index.size(0) == bs && retrive_index.size(1) == nd, "retrive_index shape");
  TORCH_CHECK(retrive_next_token.size(0) == bs && retrive_next_token.size(1) == nd, "retrive_next_token shape");
  TORCH_CHECK(retrive_next_sibling.size(0) == bs && retrive_next_sibling.size(1) == nd, "retrive_next_sibling shape");
  TORCH_CHECK(target_predict.size(0) == bs && target_predict.size(1) == nd, "target_predict shape");
  TORCH_CHECK(accept_index.size(0) == bs, "accept_index rows");
  TORCH_CHECK(accept_token_num.size(0) == bs, "accept_token_num rows");
  TORCH_CHECK(predicts.size(0) == bs * nd, "predicts length");
}

}  // namespace

// Baseline: recovered upstream kernel, grid(bs)/block(1).
void baseline_verify_tree_greedy(
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor target_predict,
    at::Tensor predicts,
    at::Tensor accept_index,
    at::Tensor accept_token_num) {
  validate(candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict,
           predicts, accept_index, accept_token_num);
  const uint32_t bs = static_cast<uint32_t>(candidates.size(0));
  const uint32_t nd = static_cast<uint32_t>(candidates.size(1));
  const uint32_t nss = static_cast<uint32_t>(accept_index.size(1));
  auto stream = at::cuda::getCurrentCUDAStream();
  baseline_vtg::launch_baseline(
      predicts.data_ptr<int32_t>(),
      accept_index.data_ptr<int32_t>(),
      accept_token_num.data_ptr<int32_t>(),
      candidates.data_ptr<int64_t>(),
      retrive_index.data_ptr<int64_t>(),
      retrive_next_token.data_ptr<int64_t>(),
      retrive_next_sibling.data_ptr<int64_t>(),
      target_predict.data_ptr<int64_t>(),
      bs, nss, nd, stream);
}

// Candidate: specialized lane-per-request kernel with baseline fallback.
void candidate_verify_tree_greedy(
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor target_predict,
    at::Tensor predicts,
    at::Tensor accept_index,
    at::Tensor accept_token_num) {
  validate(candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict,
           predicts, accept_index, accept_token_num);
  const uint32_t bs = static_cast<uint32_t>(candidates.size(0));
  const uint32_t nd = static_cast<uint32_t>(candidates.size(1));
  const uint32_t nss = static_cast<uint32_t>(accept_index.size(1));
  auto stream = at::cuda::getCurrentCUDAStream();
  candidate_vtg::dispatch_verify_tree_greedy(
      predicts.data_ptr<int32_t>(),
      accept_index.data_ptr<int32_t>(),
      accept_token_num.data_ptr<int32_t>(),
      candidates.data_ptr<int64_t>(),
      retrive_index.data_ptr<int64_t>(),
      retrive_next_token.data_ptr<int64_t>(),
      retrive_next_sibling.data_ptr<int64_t>(),
      target_predict.data_ptr<int64_t>(),
      bs, nss, nd, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("baseline_verify_tree_greedy", &baseline_verify_tree_greedy,
        "Recovered upstream verify_tree_greedy (baseline)");
  m.def("candidate_verify_tree_greedy", &candidate_verify_tree_greedy,
        "Candidate verify_tree_greedy (specialized + baseline fallback)");
}
