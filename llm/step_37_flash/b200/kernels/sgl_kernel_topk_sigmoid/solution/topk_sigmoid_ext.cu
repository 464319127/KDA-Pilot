// Local TVM-FFI ABI binding for the topk_sigmoid baseline + candidate. Compiled in ONE
// module with baseline/topk_sigmoid_baseline.cu (symmetric flags) so both sides share an
// identical export/build style and entry path. Exports:
//   topk_sigmoid_baseline        — bridges TensorView -> torch::Tensor and calls the vendored
//                                  upstream topk_sigmoid(...) (recovered baseline, verbatim kernels)
//   topk_sigmoid_candidate       — fused native-CUDA fast path on the validated contract;
//                                  falls back to the baseline for any other combination
//   topk_sigmoid_candidate_route — host-side route diagnostic (1=fast path, 0=fallback; no launch)
//   topk_sigmoid_noop            — empty-kernel launch-floor probe (candidate's grid/block)
// All outputs are destination-passing (topk_weights/topk_indices written in place);
// gating_output is read-only; every launch uses at::cuda::getCurrentCUDAStream().

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cuda_runtime.h>
#include <vector>

#include <tvm/ffi/function.h>

#include "topk_sigmoid_ext.h"
#include "topk_sigmoid_candidate.cuh"

// Forward declaration of the vendored upstream baseline
// (defined in baseline/topk_sigmoid_baseline.cu, compiled into the same module).
void topk_sigmoid(
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& gating_output,
    const bool renormalize,
    const c10::optional<torch::Tensor>& correction_bias);

namespace {

// Map a DLPack dtype to the matching at::ScalarType, preserving the REAL element size. Unsupported
// dtypes are rejected here (not silently re-tagged), so the bridge can never reinterpret storage with
// the wrong element size; the vendored baseline then applies its own dtype TORCH_CHECKs (e.g.
// correction_bias must be float32). gating_output legitimately spans fp32/fp16/bf16 (baseline-supported).
at::ScalarType dl_to_scalar_type(DLDataType d) {
  TORCH_CHECK(d.lanes == 1, "topk_sigmoid local ABI: vector dtypes (lanes>1) are unsupported");
  if (d.code == kDLFloat && d.bits == 32) return at::kFloat;
  if (d.code == kDLFloat && d.bits == 16) return at::kHalf;
  if (d.code == kDLBfloat && d.bits == 16) return at::kBFloat16;
  if (d.code == kDLInt && d.bits == 32) return at::kInt;
  if (d.code == kDLInt && d.bits == 64) return at::kLong;
  TORCH_CHECK(false, "topk_sigmoid local ABI: unsupported tensor dtype (DLPack code=",
              static_cast<int>(d.code), " bits=", static_cast<int>(d.bits), ")");
}

// Zero-copy torch::Tensor view over a TensorView's storage, preserving its REAL dtype AND strides (so
// off-domain fallback inputs are never reinterpreted with the wrong element size or logical layout).
// For calling the vendored baseline; callers .contiguous() it before the contiguous-assuming kernel.
torch::Tensor as_torch(const tvm::ffi::TensorView& t) {
  std::vector<int64_t> sizes, strides;
  sizes.reserve(t.ndim());
  strides.reserve(t.ndim());
  for (int i = 0; i < t.ndim(); ++i) {
    sizes.push_back(t.size(i));
    strides.push_back(t.stride(i));
  }
  auto opts =
      torch::TensorOptions().dtype(dl_to_scalar_type(t.dtype())).device(torch::kCUDA, tse::device_id(t));
  void* ptr = static_cast<char*>(t.data_ptr()) + t.byte_offset();
  return torch::from_blob(ptr, sizes, strides, opts);
}

// Empty kernel launched at the candidate's grid/block to bound launch overhead.
__global__ void noop_kernel() {}

// Host-side fast-path predicate (no device reads, no host sync). The fused candidate covers
// exactly the captured Step-3.7 contract: gating [N,288] fp32 contiguous CUDA, topk_weights
// [N,8] fp32, topk_indices [N,8] i32, correction_bias [288] fp32, renormalize=True. Anything
// off-domain is rejected so the dispatcher falls back to the recovered baseline.
inline bool candidate_eligible(
    const tvm::ffi::TensorView& topk_weights,
    const tvm::ffi::TensorView& topk_indices,
    const tvm::ffi::TensorView& gating_output,
    int64_t renormalize,
    const tvm::ffi::TensorView& correction_bias) {
  if (gating_output.ndim() != 2 || topk_weights.ndim() != 2 || topk_indices.ndim() != 2 ||
      correction_bias.ndim() != 1) {
    return false;
  }
  const int64_t N = gating_output.size(0);
  const int64_t E = gating_output.size(1);
  const int64_t K = topk_weights.size(1);
  const bool shapes_ok = E == tsc::kNumExperts &&
      K == tsc::kTopK && topk_weights.size(0) == N &&
      topk_indices.size(0) == N && topk_indices.size(1) == K && correction_bias.size(0) == E;
  const bool dtypes_ok = tse::is_f32(gating_output.dtype()) && tse::is_f32(topk_weights.dtype()) &&
      tse::is_i32(topk_indices.dtype()) && tse::is_f32(correction_bias.dtype());
  const bool contig_ok = tse::is_contiguous(gating_output) && tse::is_contiguous(topk_weights) &&
      tse::is_contiguous(topk_indices) && tse::is_contiguous(correction_bias);
  // All tensors must be CUDA AND on the SAME device — otherwise the single-device launch below would
  // read/write pointers from other devices (illegal access / unsynchronized peer writes on multi-GPU).
  const int dev = tse::device_id(gating_output);
  const bool device_ok = tse::is_cuda(gating_output) && tse::is_cuda(topk_weights) &&
      tse::is_cuda(topk_indices) && tse::is_cuda(correction_bias) &&
      tse::device_id(topk_weights) == dev && tse::device_id(topk_indices) == dev &&
      tse::device_id(correction_bias) == dev;
  const bool scalars_ok = renormalize != 0;  // captured contract is renormalize=True
  return shapes_ok && dtypes_ok && contig_ok && device_ok && scalars_ok;
}

}  // namespace

void topk_sigmoid_baseline(
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView topk_indices,
    tvm::ffi::TensorView gating_output,
    int64_t renormalize,
    tvm::ffi::TensorView correction_bias) {
  // The vendored baseline supports only fp32 topk_weights / int32 topk_indices and a single device
  // (it writes via data_ptr<float>()/<int>() and checks neither). Validate before any device op so
  // unsupported output dtypes / cross-device inputs raise cleanly instead of corrupting memory.
  TORCH_CHECK(tse::is_f32(topk_weights.dtype()), "topk_sigmoid: topk_weights must be float32");
  TORCH_CHECK(tse::is_i32(topk_indices.dtype()), "topk_sigmoid: topk_indices must be int32");
  const int base_dev = tse::device_id(gating_output);
  TORCH_CHECK(tse::is_cuda(gating_output) && tse::is_cuda(topk_weights) && tse::is_cuda(topk_indices) &&
                  tse::is_cuda(correction_bias) && tse::device_id(topk_weights) == base_dev &&
                  tse::device_id(topk_indices) == base_dev && tse::device_id(correction_bias) == base_dev,
              "topk_sigmoid: all tensors must be CUDA and on the same device");
  const at::cuda::OptionalCUDAGuard guard(at::Device(at::kCUDA, base_dev));
  // Strided views over the caller's storage. The vendored kernel assumes contiguous row-major, so
  // pass .contiguous() read-only inputs and scatter results back into any strided in-place output
  // (`.contiguous()` is the same tensor — no copy — when the view is already contiguous).
  torch::Tensor w_view = as_torch(topk_weights);
  torch::Tensor idx_view = as_torch(topk_indices);
  torch::Tensor gating = as_torch(gating_output).contiguous();
  torch::Tensor bias = as_torch(correction_bias).contiguous();
  torch::Tensor w = w_view.contiguous();
  torch::Tensor idx = idx_view.contiguous();
  c10::optional<torch::Tensor> bias_opt(bias);
  topk_sigmoid(w, idx, gating, renormalize != 0, bias_opt);
  if (!w_view.is_contiguous()) w_view.copy_(w);
  if (!idx_view.is_contiguous()) idx_view.copy_(idx);
}

// No-bias variants (missing-bias fallback): the captured contract always supplies a bias and
// the main ABI passes it as a required TensorView, so a `correction_bias=None` case is represented
// here by separate entrypoints used for fallback verification. The candidate has no no-bias fast
// path (it requires a bias), so candidate_nobias always routes to the baseline with c10::nullopt.
void topk_sigmoid_baseline_nobias(
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView topk_indices,
    tvm::ffi::TensorView gating_output,
    int64_t renormalize) {
  TORCH_CHECK(tse::is_f32(topk_weights.dtype()), "topk_sigmoid: topk_weights must be float32");
  TORCH_CHECK(tse::is_i32(topk_indices.dtype()), "topk_sigmoid: topk_indices must be int32");
  const int base_dev = tse::device_id(gating_output);
  TORCH_CHECK(tse::is_cuda(gating_output) && tse::is_cuda(topk_weights) && tse::is_cuda(topk_indices) &&
                  tse::device_id(topk_weights) == base_dev && tse::device_id(topk_indices) == base_dev,
              "topk_sigmoid: all tensors must be CUDA and on the same device");
  const at::cuda::OptionalCUDAGuard guard(at::Device(at::kCUDA, base_dev));
  torch::Tensor w_view = as_torch(topk_weights);
  torch::Tensor idx_view = as_torch(topk_indices);
  torch::Tensor gating = as_torch(gating_output).contiguous();
  torch::Tensor w = w_view.contiguous();
  torch::Tensor idx = idx_view.contiguous();
  topk_sigmoid(w, idx, gating, renormalize != 0, c10::nullopt);
  if (!w_view.is_contiguous()) w_view.copy_(w);
  if (!idx_view.is_contiguous()) idx_view.copy_(idx);
}

void topk_sigmoid_candidate_nobias(
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView topk_indices,
    tvm::ffi::TensorView gating_output,
    int64_t renormalize) {
  // No bias -> off the fused fast-path contract -> always fall back to the recovered baseline.
  topk_sigmoid_baseline_nobias(topk_weights, topk_indices, gating_output, renormalize);
}

int64_t topk_sigmoid_candidate_route_nobias(
    tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, int64_t) {
  return 0;  // missing bias is never on the candidate fast path
}

void topk_sigmoid_candidate(
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView topk_indices,
    tvm::ffi::TensorView gating_output,
    int64_t renormalize,
    tvm::ffi::TensorView correction_bias) {
  if (!candidate_eligible(topk_weights, topk_indices, gating_output, renormalize, correction_bias)) {
    topk_sigmoid_baseline(topk_weights, topk_indices, gating_output, renormalize, correction_bias);
    return;
  }
  const int64_t N = gating_output.size(0);
  if (N <= 0) return;
  const at::cuda::OptionalCUDAGuard guard(at::Device(at::kCUDA, tse::device_id(gating_output)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  tsc::launch_topk_sigmoid_candidate(
      tse::dptr<float>(gating_output),
      tse::dptr<float>(correction_bias),
      tse::mptr<float>(topk_weights),
      tse::mptr<int>(topk_indices),
      static_cast<int>(N),
      renormalize != 0,
      stream);
}

void topk_sigmoid_noop(tvm::ffi::TensorView gating_output) {
  const int64_t N = gating_output.size(0);
  if (N <= 0) return;
  const at::cuda::OptionalCUDAGuard guard(at::Device(at::kCUDA, tse::device_id(gating_output)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(N));
  dim3 block(tsc::kBlockThreads);
  noop_kernel<<<grid, block, 0, stream>>>();
}

int64_t topk_sigmoid_candidate_route(
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView topk_indices,
    tvm::ffi::TensorView gating_output,
    int64_t renormalize,
    tvm::ffi::TensorView correction_bias) {
  return candidate_eligible(topk_weights, topk_indices, gating_output, renormalize, correction_bias)
             ? 1
             : 0;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_sigmoid_baseline, topk_sigmoid_baseline);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_sigmoid_candidate, topk_sigmoid_candidate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_sigmoid_candidate_route, topk_sigmoid_candidate_route);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_sigmoid_noop, topk_sigmoid_noop);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_sigmoid_baseline_nobias, topk_sigmoid_baseline_nobias);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_sigmoid_candidate_nobias, topk_sigmoid_candidate_nobias);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_sigmoid_candidate_route_nobias, topk_sigmoid_candidate_route_nobias);
