# Standalone LLM Kernel Benchmark Contract

This repo optimizes SGLang **LLM serving kernels** (per model+arch, e.g.
`llm/glm_52/b200`) without interacting with a live SGLang server or checkout at
benchmark runtime. SGLang is only a source-code provider for the local baseline.

Each kernel task's `prompt.md` carries the captured Python-interface, call count,
and per-variant shapes (`docs/evidence.json` has the full records). This contract
defines how to turn those into a fair, reproducible baseline-vs-candidate result.

## Hard Rules

- Do not patch, import, monkey-patch, or install into SGLang during correctness
  or benchmark runs. All benchmark code calls only files in the task directory.
- Copy the relevant upstream SGLang kernel source into `baseline/` before
  implementing the candidate. Resolve the latest upstream SGLang `main` commit at
  baseline-recovery time, copy the kernel code from that exact commit, and record
  the repo URL, branch (`main`), resolved commit, resolution time, and copied
  file list in `docs/baseline_source.md`.
- Both sides expose **matching local entry points / ABI**. Any wrapper overhead
  included for one side must be included for the other. Do not time a copied
  baseline through a heavy Python/JIT wrapper while timing the candidate through
  a lean direct path.
- The promoted candidate is **native CUDA** (see `llm_kernel_optimization_rules.md`
  → Implementation Language). If the copied baseline is Triton/CuTe-DSL/Python,
  keep it in `baseline/` and build a local adapter with the same call signature,
  argument order, stream behavior, and output-allocation policy as the candidate.
- Every CUDA launch must use PyTorch's current stream
  (`at::cuda::getCurrentCUDAStream()`).
- Compile flags must be symmetric between baseline and candidate whenever they
  can change numerics or codegen. No one-sided `--use_fast_math`. Record all
  flags in `docs/benchmark_method.md`.
- Workloads are frozen before tuning. Changing workloads, tolerances, scoring, or
  timing rules requires deleting old results and remeasuring both sides.
- Preserve and respect the captured **input contract**: dtype, device, and
  **`is_contiguous`/stride** (most GLM-5.2 captures are `contiguous=False`),
  plus scalar kwargs (e.g. `is_neox`, `rope_dim`, `eps`, group size). A candidate
  that only handles contiguous inputs but is benchmarked on strided production
  shapes is invalid; either handle strides or fall back to baseline for them.

## Required Directory Contents After First Agent Milestone

```text
baseline/    copied upstream source + local callable exposing the baseline ABI
solution/    candidate .cu/.cuh/.cpp/.h + local callable exposing the same ABI
bench/
  workloads.json      frozen workloads (source of truth, see Workload Rules)
  benchmark.py        copied from ../../../docs/standalone_llm_benchmark_template.py
  adapter.py          make_case / call_baseline / call_candidate
  correctness.py      oracle + tolerance checks (see llm_correctness_contract.md)
  results.jsonl       raw per-run records
docs/
  baseline_source.md  upstream URL/branch/commit/time/files
  benchmark_method.md  compile flags, build paths, timing settings
  results.md          per-shape baseline-vs-candidate table + headline + conclusion
  dispatch.md         only when shape-specialized (bucket -> kernel + per-bucket speedup)
  run_log.md          host/GPU id/model + before/after idle state
config.toml
```

`bench/benchmark.py` must start from
[`standalone_llm_benchmark_template.py`](standalone_llm_benchmark_template.py).
Do not invent a different timing harness unless this template has a documented
bug and both sides are remeasured after the fix.

## ABI Pattern

Prefer the local direct-symbol CUDA ABI for both sides:
`TVM_FFI_DLL_EXPORT_TYPED_FUNC`, `tvm::ffi::TensorView` arguments, output tensors
passed last, `destination_passing_style = true`, launch on
`at::cuda::getCurrentCUDAStream()`. One exported function per upstream public
entry point, or a single exported function with an explicit selector argument —
baseline and candidate must make the same choice.

## Workload Rules

- `bench/workloads.json` is the source of truth and must be populated from the
  task's captured variants (`prompt.md` + `docs/evidence.json`) before tuning.
- Deduplicate the captured variants into distinct (shape, dtype, stride, scalar)
  rows. Tag each row `production: true/false` and mark which rows are in the
  headline score (the high-call-count production rows) vs regression-only edge
  rows. Do not silently drop a high-frequency captured shape.
- Each workload records function/selector, tensor shapes, dtypes, **strides**,
  scalar parameters, tolerance, random seed, and headline inclusion.
- Add a small regression grid for edge shapes/dtypes from
  `llm_correctness_contract.md`.
- Any missing baseline, missing candidate, compile failure, runtime failure, or
  correctness failure on a workload makes the whole benchmark invalid. No silent
  skips.

## Timing Rules

- Run each workload in an isolated subprocess when possible.
- Fresh random inputs per trial; tensors stable inside a trial, changing between
  trials.
- Preallocate output tensors before timing. The timed region excludes input
  generation, Python setup, JIT build, imports, allocation, and data restore.
- Warm both sides before measurement.
- CUDA events for GPU time; wrapper-inclusive wall-clock only as a secondary
  diagnostic.
- Inner-loop amplification: record N back-to-back invocations inside one event
  pair and divide by N. Increase N until the sample is at least ~1000us or N hits
  the cap. (Critical for the small/cheap elementwise kernels here, where a single
  launch is dominated by launch overhead.)
- Interleaved A/B sampling per trial (baseline, candidate, ... or a deterministic
  reverse order) to cancel clock/thermal drift.
- Report median, mean, std, min, p10, p90 for both sides on every workload.
- Per-workload speedup = `baseline_median_us / candidate_median_us`.
- Headline = equal-weight geometric mean of speedup over the production
  workloads. Also report arithmetic mean as a secondary metric.

Recommended defaults (also mirrored in each task `config.toml`):

```toml
[benchmark]
warmup_runs = 10
iterations = 200
num_trials = 7
inner_iterations_min = 1
inner_iterations_max = 4096
target_sample_us = 1000
timeout_seconds = 600
use_isolated_runner = true
```

## Provenance

Every benchmark result must record: task slug + target GPU; upstream baseline
commit + copied files; candidate source hash; exact command; CUDA, PyTorch,
compiler, and TVM-FFI versions; GPU model/id and idle state before/after;
workload count and trial/iteration/inner-loop settings; correctness summary. Do
not keep benchmark numbers without this provenance.
