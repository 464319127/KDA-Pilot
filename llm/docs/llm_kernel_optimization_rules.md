# LLM Kernel Optimization Rules

These rules apply to every standalone LLM serving-kernel task (e.g.
`llm/glm_52/b200/kernels/*`). They keep each task `prompt.md` short while
preserving the guardrails. Read this with `standalone_llm_benchmark.md` and
`llm_correctness_contract.md` before optimizing.

## Baseline And Candidate Pairing

Each task ends with two local implementations:

- `baseline/`: copied upstream SGLang kernel source + a local callable exposed
  through the task benchmark ABI.
- `solution/`: the optimized candidate exposed through the **exact same** ABI.

Before copying, resolve the latest commit on upstream SGLang `main` and copy the
kernel source from that exact commit. Do not use a stale pinned commit, a local
production checkout with unknown drift, or kernel code from a previous task.
Record repo URL, branch (`main`), resolved commit SHA, resolution time, and
copied files in `docs/baseline_source.md`.

If the baseline is CUDA/C++, baseline and candidate must use the same
registration/export/build style (same `TVM_FFI_DLL_EXPORT_TYPED_FUNC` pattern,
output tensors last, etc.). If the baseline is Triton/CuTe-DSL/Python, keep it in
`baseline/` and build a local adapter with the same signature, argument order,
stream behavior, and output-allocation policy as the candidate adapter.

Every CUDA launch must use `at::cuda::getCurrentCUDAStream()`.

## Implementation Language (CUDA only)

Iterate and implement the candidate in **native CUDA**. The promoted candidate
must be built from workspace-owned C++/CUDA source (`.cu`/`.cuh`/`.cpp`/`.h`)
compiled with nvcc or an equivalent CUDA extension build (CUTLASS / CuTe C++
templates are allowed — they are CUDA C++). Do **not** use Triton, TileLang,
CuTe-DSL (Python `cute.compile`), `torch.compile`, or any other DSL / prebuilt op
as the candidate execution path. Such sources and the upstream kernel may be
studied or ported into workspace CUDA, but the benchmarked/promoted candidate is
the native CUDA implementation. Python is allowed only for harnesses, bindings,
benchmark scripts, and dispatch glue.

## Compile Flags

Compile flags must be symmetric whenever they can change numerics or codegen. No
one-sided `--use_fast_math` (default: no fast math). Do not add architecture or
math-mode flags to only one side. Avoid comparing different builder paths (e.g.
`torch.utils.cpp_extension` on one side, direct registration on the other).
Record all flags in `docs/benchmark_method.md`. PDL may be tested only if the
upstream kernel has a comparable path and it wins on the task's real workloads.

## Remote GPU Rule

Validate and benchmark on the task's target arch (B200 tasks on B200). Before GPU
work, inspect `nvidia-smi`, pick a GPU with no active compute and no meaningful
memory occupancy, export it as `REMOTE_GPU_ID`, and use it consistently for
baseline, candidate, correctness, benchmark, profiling, and NCU in that run.
Record host, GPU id, GPU model, and before/after state in `docs/run_log.md`. Use
a task-owned remote workspace; never write into another task's workspace.

(Launchers pin a round-robin `REMOTE_GPU_ID` per kernel; honor it unless it is
busy, then wait/retry or ask.)

## Correctness Before Performance

Before optimization, recover: the upstream baseline source; the public callable
arguments and scalar parameters; the production workload rows (from the captured
variants); and the regression grid + tolerances from
`llm_correctness_contract.md`. The final candidate must pass both the production
workload checks and the regression grid before any benchmark result counts.
Preserve explicit NaN/Inf checks and poison output buffers before each run.

## Benchmark And Evidence

Use `standalone_llm_benchmark_template.py` as the timing harness. Do not change
workloads, tolerances, scoring, or timing rules after tuning starts unless both
sides are remeasured. Every RLCR iteration refreshes its context (this doc, the
prompt, current benchmark/profile evidence, `external/KernelWiki/SKILL.md`,
`external/ncu-report-skill/SKILL.md`, and the warp-spec timeline for
warp-specialized candidates) before the next edit/run/no-go.

A final performance claim reports: median/mean/std/min/p10/p90 latency per
workload; equal-weight geometric-mean speedup over production workloads; exact
commands; baseline commit + candidate hash; GPU host/id/model + idle evidence. A
final win or no-go includes a roofline-style explanation: estimated bytes moved,
useful ops, achieved bandwidth and/or FLOP/s when relevant, and the active bound
or blocker. Do not finalize a no-go just because the first candidate loses — a
no-go needs baseline numbers, ≥1 reasoned candidate attempt, correctness status,
benchmark/NCU evidence, and a named active bound.

## Shape Specialization

Shape-specialized kernels, template variants, and dispatchers are expected when
evidence shows different workload buckets need different block sizes, vector
widths, layouts, or register tradeoffs (the captured interfaces here span wide
shape ranges). When specialized, write `docs/dispatch.md` with the bucket
condition, the selected entry points, per-bucket latency/speedup, and the reason.
Any shape/parameter combo not covered by a specialized path must fall back to the
recovered baseline. Keep dispatch cheap (no host syncs on the hot path).

## PR Scope

The final PR includes only: kernel source for the copied baseline, optimized
solution, local ABI, benchmark adapter, and correctness/benchmark harness; the
per-shape baseline-vs-candidate comparison + conclusion (normally
`docs/results.md` + `docs/dispatch.md`); and small method/provenance notes. Do
not commit raw NCU reports, Nsight traces, profiler directories, warp-timeline
dumps, temporary binaries, build outputs, scratch logs, failed-experiment dumps,
or large benchmark JSONL unless explicitly requested. Keep those local, leave
them unstaged before opening the PR.

## Completion Bar

A task is complete only when: `baseline/`, `solution/`, `bench/`, and `docs/`
hold the required artifacts; production-workload correctness passes; the
regression grid passes; the benchmark uses the standard timing rules; NCU or a
clear roofline analysis explains the result or blocker; `docs/results.md`
summarizes the final command, per-shape comparison, result, and conclusion; and
the staged PR diff excludes raw profiling/NCU/build/scratch artifacts.
