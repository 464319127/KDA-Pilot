# Baseline Source Provenance

## Upstream
- Repository: https://github.com/sgl-project/sglang
- Branch: `main`
- Resolved commit SHA: `7e6587c94a1d0305815a14067c5d3cc02a9b0f36`
- Resolution method: `git ls-remote https://github.com/sgl-project/sglang main` (this commit was `main` HEAD at resolution time)
- Resolution time: 2026-06-23 (UTC)

## Interface recovered
- Python interface: `sgl_kernel.build_tree_kernel_efficient`
- Torch op schema (from `sgl-kernel/csrc/common_extension.cc`):
  `build_tree_kernel_efficient(Tensor parent_list, Tensor selected_index, Tensor verified_seq_len, Tensor tree_mask, Tensor positions, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, int topk, int depth, int draft_token_num, int tree_mask_mode) -> ()`
  Registered CUDA impl: `m.impl("build_tree_kernel_efficient", torch::kCUDA, &build_tree_kernel_efficient)`.
- Semantics: returns `None`; mutates `tree_mask`, `positions`, `retrive_index`, `retrive_next_token`, `retrive_next_sibling` in place; launches on `at::cuda::getCurrentCUDAStream()`.

## Files inspected upstream (for recovery / contract)
- `sgl-kernel/csrc/speculative/eagle_utils.cu` — the op + device kernels (source of the copied baseline).
- `sgl-kernel/csrc/common_extension.cc` — op registration / schema.
- `sgl-kernel/include/sgl_kernel_ops.h` — C++ declaration.
- `sgl-kernel/python/sgl_kernel/speculative.py` — thin Python wrapper.
- `python/sglang/srt/speculative/eagle_utils.py` — the callsite that sets output **pre-state** (FULL_MASK: `tree_mask = full(..., True)`; `retrieve_buf = full((3, bs, num_verify), -1)`; `positions = empty(bs*num_verify)`).

## Copied into this workspace
- `baseline/build_tree_baseline.cu` — verbatim copy of the two device kernels
  `build_tree_efficient` and `build_tree_efficient_partial_packed` from
  `eagle_utils.cu`, plus the upstream `build_tree_kernel_efficient` host body
  **renamed** `build_tree_baseline` (no logic change) so the candidate can be
  exposed through the identical local ABI.
- The unrelated `verify_tree_greedy` op (its own kernel task,
  `sgl_kernel_verify_tree_greedy`) and the `pytorch_extension_utils.h` include it
  requires are intentionally **not** copied; `build_tree` uses neither.

## Local edits vs upstream
- Device kernel bodies (`build_tree_efficient`, `build_tree_efficient_partial_packed`)
  are byte-identical to upstream.
- The host entry point is renamed (`build_tree_kernel_efficient` -> `build_tree_baseline`),
  wrapped in an anonymous `namespace` (avoid symbol collisions when compiled with the
  candidate), and adapted to the local TVM-FFI direct-symbol ABI: it takes
  `tvm::ffi::TensorView` args + `int64_t` scalars (instead of `at::Tensor`/`int64_t`),
  extracts data pointers / sizes via the `bte::` helpers, launches on
  `at::cuda::getCurrentCUDAStream()`, and is exported with
  `TVM_FFI_DLL_EXPORT_TYPED_FUNC`. No logic change.

## Local ABI
- Both baseline and candidate are exposed through ONE TVM-FFI direct-symbol module
  (`TVM_FFI_DLL_EXPORT_TYPED_FUNC` / `tvm::ffi::TensorView`, destination-passing,
  `at::cuda::getCurrentCUDAStream()`), built together by `bench/build_ext.py` via
  `tvm_ffi.cpp.load` with symmetric flags — matching the repo's `diffusion/kernels/*`
  pattern. Identical registration/export/build/call path for both sides. See
  `docs/benchmark_method.md`.
