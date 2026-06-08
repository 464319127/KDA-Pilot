# KDA Prompt: b200_int8_scaled_mm__m64_n2048_k2048_bias

Develop an optimized CUDA/C++ or CUDA inline-PTX kernel for SGLang's
`int8_scaled_mm` focused shape while preserving numerical correctness. The
target machine is NVIDIA B200. This prompt is intended for the Kernel Design
Agents workflow with official Humanize, KernelWiki, and ncu-report-skill
available.

This is a single-kernel KDA prompt in the style of
`BBuf/kernel-design-agents`: first recover the exact baseline and correctness
contract, then run profiling-guided optimization, then specialize only if the
measured shape evidence justifies it.

## Kernel Information

- Kernel folder: `b200_int8_scaled_mm__m64_n2048_k2048_bias`
- Source project: SGLang
- Operation type: scaled int8 matrix multiply with fp16 output and fused bias
- Hardware target: NVIDIA B200
- Focused shape:
  - `M=64`
  - `N=2048`
  - `K=2048`
  - `out_dtype=fp16`
  - `bias=true`
- Baseline: current SGLang implementation path for `int8_scaled_mm`
- Promotion target: at least `2.5x` faster than the SGLang baseline median
  latency for the exact same shape, dtype, layout, bias behavior, and selected
  idle B200 GPU environment.

## Environment And Remote Rule

Use the `ion-b200` remote GPU environment for all B200 work. All CUDA, Python,
pip, nvcc, build, test, benchmark, and Nsight Compute commands must run inside
the existing `sglang_bbuf` Docker container on `ion-b200`, with an idle B200
GPU selected.

Before running GPU work, inspect `nvidia-smi` and choose a GPU with no active
compute processes and no meaningful memory occupancy. Export that id as
`REMOTE_GPU_ID` and use it consistently for the baseline, candidate,
benchmark, profiler, and NCU commands in the current run.

Use this command pattern for remote execution:

```bash
ssh ion-b200 'REMOTE_GPU_ID=<idle-gpu-id>; docker exec sglang_bbuf bash -lc "CUDA_VISIBLE_DEVICES=${REMOTE_GPU_ID} <command>"'
```

Do not run Python, pip, nvcc, builds, tests, benchmarks, or profiling directly
on the `ion-b200` host.

When multiple sessions share the same remote container, create a task-owned
remote workspace under:

```text
/home/sglang-omni/bbuf/kda_runs/b200_int8_scaled_mm__m64_n2048_k2048_bias/<timestamp-or-session-id>
```

Export it as `REMOTE_KDA_DIR`. Keep build outputs, benchmark logs, profiler
traces, NCU reports, and captured tensors inside that directory. Do not reuse
shared temporary validation paths across tasks.

Before and after every benchmark or profile run, check GPU state and record the
selected host, GPU id, and GPU model in `benchmark.csv` or `docs/draft.md`.

## Workspace Rule

All work for this kernel must happen inside the current prompt folder, the
folder that contains this `prompt.md`.

Required local files:

- `src/`: optimized wrapper, CUDA/C++ sources, inline-PTX helpers, build glue,
  dispatcher code, and registration entrypoint.
- `tests/`: isolated correctness tests and optional SGLang integration smoke
  tests.
- `docs/draft.md`: implementation-plan draft written before code changes.
- `benchmark.csv`: every measured baseline and candidate comparison.
- `solutions.jsonl`: every candidate implementation, parent link, status, and
  evidence pointer.
- `profile/`: profiler traces and summaries.
- `ncu/`: Nsight Compute reports when collected.

Keep SGLang checkouts as dependencies to inspect, run, or patch temporarily.
The optimized implementation and evidence for this task stay local to this
folder unless intentionally promoted later.

## Prior Art Research Scope

Before choosing an implementation strategy, inspect SGLang, CUTLASS/CuTe,
CUDA samples, PyTorch, vLLM, TensorRT-LLM, DeepGEMM, and public Blackwell INT8
GEMM kernels for directly relevant ideas. Use KernelWiki when prior B200,
SM100, CUTLASS, SGLang, or int8 GEMM evidence can guide a design choice.

Record reviewed source paths, commits or installed versions, and which ideas
were kept or rejected in `docs/draft.md` and `solutions.jsonl`.

## Baseline Recovery And Correctness Harness

Recover the baseline before writing optimized code:

1. Inspect the current SGLang implementation path for `int8_scaled_mm`.
2. Build a reproducible baseline harness for the exact focused case.
3. Capture or generate baseline inputs covering the expected int8 layouts,
   scales, fp16 output, and bias behavior.
4. Fill `tests/test_correctness.py` before optimization. The test must compare
   candidate output against the current SGLang result and a PyTorch reference
   when practical.
5. Preserve explicit NaN/Inf checks in every validator.

Correctness reporting must include max absolute error, relative error, and the
tolerance used. The final candidate must pass correctness before benchmark
claims count.

## Benchmark Requirements

- Use warmup and repeated timing.
- Report median latency, mean latency, std, min, p10, and p90.
- Compare every candidate against the SGLang baseline from the same selected
  idle B200 GPU and container environment.
- Keep benchmark scripts and raw result logs in this folder.
- Every claimed improvement must identify the candidate commit or file version
  and the exact command used to produce the result.
- Use Nsight Compute when a correct candidate is not clearly target-complete or
  when profiler evidence would change the next edit.

## Optimization Exploration Policy

Use the KDA-style exploration loop:

- list candidate optimization directions in `docs/draft.md`;
- rank them by expected benefit, implementation risk, and how directly they
  attack the measured bottleneck;
- try each direction for a bounded number of focused iterations;
- keep, revise, or reject each direction with correctness, benchmark, and NCU
  evidence;
- maintain parent links in `solutions.jsonl` so later runs can reconstruct the
  search DAG.

Consider B200/SM100-specific paths such as `tcgen05` INT8 MMA, TMEM/TMA where
appropriate, warp specialization, persistent scheduling, Stream-K or split-K,
cluster shape choices, vectorized loads/stores, shared-memory staging, and a
fused bias/output epilogue.

## Shape Specialization Policy

Keep the first optimization focused on the exact target shape. If profiler or
benchmark evidence reveals nearby regimes that need different kernels, record
them as follow-up shape buckets rather than changing this task's acceptance
target. A dispatcher is allowed only when it preserves the focused-shape target
and has a safe fallback to the SGLang baseline.

## Interface Contract

Add the candidate under `src/` and expose:

```text
src/register.py
```

with:

```python
def optimized_wrapper(*args, **kwargs):
    ...

def register() -> dict:
    ...
```

`optimized_wrapper` must preserve the recovered SGLang callsite contract and
fall back to the baseline implementation for unsupported shapes, dtypes,
layouts, devices, or bias configurations. See `interface.md`.

## Required Workflow

1. Confirm the current directory is this kernel folder.
2. Read `../../external/KernelWiki/SKILL.md` and
   `../../external/ncu-report-skill/SKILL.md` from this kernel folder.
3. Recover the SGLang baseline path, tensor contract, and exact benchmark
   command.
4. Write an implementation-plan draft to `docs/draft.md`.
5. Run official Humanize plan generation on that draft.
6. Start official Humanize RLCR from this kernel folder.
7. Do not implement kernels, run long benchmarks, or collect NCU evidence before
   RLCR is active.
8. Record every candidate in `solutions.jsonl` and every performance result in
   `benchmark.csv`.

## Completion Bar

The work is complete only when:

- correctness tests pass for the focused shape with bias;
- B200 benchmark evidence shows at least `2.5x` median-latency speedup over the
  current SGLang baseline, or a well-supported no-go conclusion explains why no
  defensible path remains under the available workspace;
- NCU or benchmark evidence explains the active bottleneck and final design;
- `prompt.md`, `interface.md`, `benchmark.csv`, and `solutions.jsonl` are updated
  with the final result.
