# Benchmark Method — `verify_tree_greedy`

## Build / ABI

Both the recovered baseline and the candidate are compiled into a **single** TVM-FFI
module (`bench/verify_tree_greedy_ffi.cu`) built with `tvm_ffi.cpp.load` (tvm_ffi 0.1.9).
Both entry points are exported via `TVM_FFI_DLL_EXPORT_TYPED_FUNC` and take
`tvm::ffi::TensorView` arguments with the **identical** signature:

- `baseline_verify_tree_greedy(candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict, predicts, accept_index, accept_token_num)`
- `candidate_verify_tree_greedy(... same args ...)`

Both run the same host-side validation, allocate nothing (the three outputs are passed in,
preallocated by the harness), and launch on the CUDA stream returned by the TVM-FFI
environment (`TVMFFIEnvGetStream`), which the harness binds to PyTorch's current stream via
`tvm_ffi.use_torch_stream()` — the same stream the benchmark's CUDA events are recorded on.

### Argument order — outputs LAST (documented remap)

Upstream `sgl_kernel.verify_tree_greedy` lists the three mutable outputs FIRST:
`(predicts, accept_index, accept_token_num, candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict)`.
The local ABI passes the five inputs first and the three mutable outputs **last**,
for both sides identically:
`(candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict | predicts, accept_index, accept_token_num)`.
The internal launch wrappers remap back to the upstream kernel order.

### ABI mechanism

This is the literal local ABI required by AC-2 and `llm_kernel_optimization_rules.md`: a
TVM-FFI direct-symbol typed export (`TVM_FFI_DLL_EXPORT_TYPED_FUNC`) with
`tvm::ffi::TensorView` arguments, outputs passed last, destination-passing style (the
harness preallocates the three outputs), shared validation, and a single shared build for
both sides. The Python harness loads the module with `tvm_ffi.cpp.load` and calls the two
exported symbols; `bench/adapter.py`'s `call_baseline` / `call_candidate` differ only by
the exported symbol name, so the comparison is symmetric.

Note: upstream sgl-kernel registers `verify_tree_greedy` via `TORCH_LIBRARY` with
`at::Tensor` (not TVM-FFI). The TVM-FFI direct-symbol pattern is the KDA-Pilot-mandated
*local* benchmark ABI, provided by the `tvm-ffi` package (headers under
`tvm_ffi/include` + `libtvm_ffi.so`). The kernel sources in `baseline/` and `solution/`
are ABI-agnostic (plain `cudaStream_t` launchers); only this binding depends on TVM-FFI.

## Compile flags (symmetric)

- C++: `-O3`
- CUDA: `extra_cuda_cflags=["-O3"]` (no `--use_fast_math`; no one-sided arch/math-mode flags)
- Architecture: `TORCH_CUDA_ARCH_LIST=10.0` for the build GPU (NVIDIA B200 = `sm_100`); `tvm_ffi.cpp.load` invokes `nvcc`.
- Both kernels live in the same module/translation unit, so flags are identical by construction.

These kernels do only integer comparisons and int loads/stores (no floating point),
so fast-math is irrelevant to numerics; correctness is exact integer/structural match.

## Timing policy

`bench/benchmark.py` is byte-identical to `llm/docs/standalone_llm_benchmark_template.py`
(verified). Defaults used (also in `config.toml`):

| setting | value |
|---|---|
| warmup_runs | 10 |
| num_trials | 7 |
| inner_iterations_min / max | 1 / 4096 |
| target_sample_us | 1000 |
| isolated subprocess | true |

- **Inner-loop amplification** is essential: a single launch is sub-microsecond, so the
  harness ramps N back-to-back launches per CUDA-event pair until the sample ≥ ~1000µs
  (or N hits 4096), then divides by N.
- **Interleaved A/B** ordering per trial cancels clock/thermal drift.
- **CUDA-event GPU time is the authoritative metric** (DEC-2); launch-inclusive
  wall-clock is recorded as a secondary diagnostic only.
- Headline = equal-weight geometric mean of `baseline_median_us / candidate_median_us`
  over the `production` rows (computed by the harness).

## Device normalization

Captured device ids span `cuda:0..7` (an 8-way TP capture). For benchmarking, the task is
pinned to physical GPU 7: run with `CUDA_VISIBLE_DEVICES=7` and `--device cuda:0` (so
`cuda:0` maps to physical GPU 7), or `--device cuda:7`. Capture device ids are not encoded
as separate workloads.

## Provenance (filled after the remote run)

To be recorded in `docs/run_log.md` and `docs/results.md`: host, GPU id/model, before/after
idle state, CUDA / PyTorch / compiler versions, exact commands, baseline commit
(`7e6587c94a1d0305815a14067c5d3cc02a9b0f36`), candidate source hash, and the per-shape stats.
