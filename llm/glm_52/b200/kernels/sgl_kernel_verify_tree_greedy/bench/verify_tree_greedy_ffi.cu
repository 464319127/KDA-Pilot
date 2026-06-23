// Shared low-overhead local ABI for the standalone baseline-vs-candidate harness,
// using the KDA-Pilot-mandated TVM-FFI direct-symbol pattern (per
// llm_kernel_optimization_rules.md and standalone_llm_benchmark.md "ABI Pattern").
//
// Both entry points are exported with TVM_FFI_DLL_EXPORT_TYPED_FUNC and take
// tvm::ffi::TensorView arguments with the IDENTICAL signature: the five int64 inputs
// first, the three mutable int32 outputs LAST. Each launches on the stream provided by
// the TVM-FFI environment (TVMFFIEnvGetStream), which the harness binds to PyTorch's
// current CUDA stream via tvm_ffi.use_torch_stream(); the benchmark's CUDA events are
// recorded on that same stream. Both sides share validation and compile flags, so the
// CUDA-event-timed comparison is fair. Upstream's public signature lists the three
// outputs FIRST; the outputs-first -> outputs-last remap is documented in
// docs/benchmark_method.md.
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "verify_tree_greedy_kernel.cuh"     // baseline_vtg::launch_baseline      (-I baseline)
#include "verify_tree_greedy_candidate.cuh"  // candidate_vtg::dispatch_...        (-I solution)

namespace {

using tvm::ffi::TensorView;

inline cudaStream_t get_stream(DLDevice device) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
}

inline void require(bool cond, const char* msg) {
  if (!cond) throw std::runtime_error(std::string("verify_tree_greedy ABI: ") + msg);
}

// Shared int-tensor contract check (CUDA, dtype int<bits>, rank ndim). Exceptions are
// caught by the TVM_FFI_DLL_EXPORT_TYPED_FUNC wrapper and surfaced to Python.
inline void check_int(const TensorView& t, int bits, int ndim, const char* name) {
  require(t.device().device_type == kDLCUDA, name);
  require(t.ndim() == ndim, name);
  DLDataType dt = t.dtype();
  require(dt.code == kDLInt && dt.bits == bits && dt.lanes == 1, name);
}

// Validate the shared ABI shape/dtype contract for both sides identically.
void validate(
    const TensorView& candidates,
    const TensorView& retrive_index,
    const TensorView& retrive_next_token,
    const TensorView& retrive_next_sibling,
    const TensorView& target_predict,
    const TensorView& predicts,
    const TensorView& accept_index,
    const TensorView& accept_token_num) {
  check_int(candidates, 64, 2, "candidates");
  check_int(retrive_index, 64, 2, "retrive_index");
  check_int(retrive_next_token, 64, 2, "retrive_next_token");
  check_int(retrive_next_sibling, 64, 2, "retrive_next_sibling");
  check_int(target_predict, 64, 2, "target_predict");
  check_int(predicts, 32, 1, "predicts");
  check_int(accept_index, 32, 2, "accept_index");
  check_int(accept_token_num, 32, 1, "accept_token_num");
  const int64_t bs = candidates.shape()[0];
  const int64_t nd = candidates.shape()[1];
  require(retrive_index.shape()[0] == bs && retrive_index.shape()[1] == nd, "retrive_index shape");
  require(retrive_next_token.shape()[0] == bs && retrive_next_token.shape()[1] == nd, "retrive_next_token shape");
  require(retrive_next_sibling.shape()[0] == bs && retrive_next_sibling.shape()[1] == nd, "retrive_next_sibling shape");
  require(target_predict.shape()[0] == bs && target_predict.shape()[1] == nd, "target_predict shape");
  require(accept_index.shape()[0] == bs, "accept_index rows");
  require(accept_token_num.shape()[0] == bs, "accept_token_num rows");
  require(predicts.shape()[0] == bs * nd, "predicts length");
}

// Shared entry body: validate, derive dims, resolve the stream, cast pointers, and launch.
// The two exported functions differ only by the launcher passed in — `launch_baseline` and
// `dispatch_verify_tree_greedy` have the identical pointer/dim/stream signature.
template <typename Launch>
inline void run_verify(
    Launch launch,
    TensorView candidates,
    TensorView retrive_index,
    TensorView retrive_next_token,
    TensorView retrive_next_sibling,
    TensorView target_predict,
    TensorView predicts,
    TensorView accept_index,
    TensorView accept_token_num) {
  validate(candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict,
           predicts, accept_index, accept_token_num);
  const uint32_t bs = static_cast<uint32_t>(candidates.shape()[0]);
  const uint32_t nd = static_cast<uint32_t>(candidates.shape()[1]);
  const uint32_t nss = static_cast<uint32_t>(accept_index.shape()[1]);
  cudaStream_t stream = get_stream(predicts.device());
  launch(
      static_cast<int32_t*>(predicts.data_ptr()),
      static_cast<int32_t*>(accept_index.data_ptr()),
      static_cast<int32_t*>(accept_token_num.data_ptr()),
      static_cast<const int64_t*>(candidates.data_ptr()),
      static_cast<const int64_t*>(retrive_index.data_ptr()),
      static_cast<const int64_t*>(retrive_next_token.data_ptr()),
      static_cast<const int64_t*>(retrive_next_sibling.data_ptr()),
      static_cast<const int64_t*>(target_predict.data_ptr()),
      bs, nss, nd, stream);
}

}  // namespace

// Baseline: recovered upstream kernel, grid(bs)/block(1).
void baseline_verify_tree_greedy(
    TensorView candidates,
    TensorView retrive_index,
    TensorView retrive_next_token,
    TensorView retrive_next_sibling,
    TensorView target_predict,
    TensorView predicts,
    TensorView accept_index,
    TensorView accept_token_num) {
  run_verify(baseline_vtg::launch_baseline, candidates, retrive_index, retrive_next_token,
             retrive_next_sibling, target_predict, predicts, accept_index, accept_token_num);
}

// Candidate: specialized lane-per-request kernel with baseline fallback.
void candidate_verify_tree_greedy(
    TensorView candidates,
    TensorView retrive_index,
    TensorView retrive_next_token,
    TensorView retrive_next_sibling,
    TensorView target_predict,
    TensorView predicts,
    TensorView accept_index,
    TensorView accept_token_num) {
  run_verify(candidate_vtg::dispatch_verify_tree_greedy, candidates, retrive_index,
             retrive_next_token, retrive_next_sibling, target_predict, predicts, accept_index,
             accept_token_num);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(baseline_verify_tree_greedy, &baseline_verify_tree_greedy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(candidate_verify_tree_greedy, &candidate_verify_tree_greedy);
