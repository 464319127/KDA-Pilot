# Benchmark Method — `verify_tree_greedy`

## Build / ABI

Both the recovered baseline and the candidate are compiled into a **single** torch
CUDA extension (`bench/verify_tree_greedy_ext.cu`) built with
`torch.utils.cpp_extension.load`. The extension exports two functions with the
**identical** signature:

- `baseline_verify_tree_greedy(candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict, predicts, accept_index, accept_token_num)`
- `candidate_verify_tree_greedy(... same args ...)`

Both run the same host-side validation, launch on `at::cuda::getCurrentCUDAStream()`,
and allocate nothing (the three outputs are passed in, preallocated by the harness).

### Argument order — outputs LAST (documented remap)

Upstream `sgl_kernel.verify_tree_greedy` lists the three mutable outputs FIRST:
`(predicts, accept_index, accept_token_num, candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict)`.
The local ABI passes the five inputs first and the three mutable outputs **last**,
for both sides identically:
`(candidates, retrive_index, retrive_next_token, retrive_next_sibling, target_predict | predicts, accept_index, accept_token_num)`.
The internal launch wrappers remap back to the upstream kernel order.

### ABI mechanism note (deviation from the plan's literal wording)

The plan's AC-2 names "TVM-FFI direct-symbol typed export with `tvm::ffi::TensorView`
arguments". This task instead uses a single **symmetric torch `cpp_extension`** (pybind)
for BOTH sides. The benchmark contract only *prefers* the TVM-FFI direct-symbol ABI;
its controlling fairness rules are all satisfied here: identical signature / argument
order / output-allocation policy, the same `at::cuda::getCurrentCUDAStream()`, the same
builder path and compile flags, and no heavy one-sided wrapper. Because the kernel time
is measured with CUDA events that bracket the launch, the (symmetric) pybind dispatch
cost sits outside the timed region and cannot bias the comparison. This avoids adding a
`tvm-ffi` runtime dependency to the standalone harness while preserving low, symmetric
call overhead. (Logged in the goal-tracker Plan Evolution Log.)

## Compile flags (symmetric)

- C++: `-O3`
- CUDA: `-O3` (no `--use_fast_math`; no one-sided arch/math-mode flags)
- Architecture: auto-selected by `cpp_extension` for the build GPU (NVIDIA B200 = `sm_100`).
- Both kernels live in the same translation-unit/extension, so flags are identical by construction.

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
