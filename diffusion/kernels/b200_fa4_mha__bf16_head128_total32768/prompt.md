# KDA Prompt: b200_fa4_mha__bf16_head128_total32768

Develop a FlashAttention-4-comparable BF16 forward-only MHA kernel for NVIDIA
B200 while preserving standard scaled dot-product attention semantics. This
prompt is intended for the Kernel Design Agents workflow with official
Humanize, KernelWiki, and ncu-report-skill available.

This is a single-kernel KDA prompt in the style of
`BBuf/kernel-design-agents`: first recover a correct baseline-facing
implementation, then run profiling-guided optimization exploration, then
specialize for the configured workload shapes when specialization is justified.

## Kernel Information

- Kernel folder: `b200_fa4_mha__bf16_head128_total32768`
- Operation type: dense multi-head attention forward
- Baseline solution: official FlashAttention-4 installed in the same
  `ion-b200` container
- Hardware target: NVIDIA B200
- dtype: BF16
- head_dim: 128
- num_heads: 16
- total tokens: 32768
- Scope:
  - forward pass only
  - no backward
  - no GQA
  - no serving or framework integration
- Promotion target: beat official FlashAttention-4 by at least `5%`
  geometric-mean TFLOPS across all configured B200 cases.

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
on the `ion-b200` host. Do not pip install `flash-attn` on the host. The
container already has FlashAttention-4 installed and it is the main performance
baseline.

When multiple sessions share the same remote container, create a task-owned
remote workspace under:

```text
/home/sglang-omni/bbuf/kda_runs/b200_fa4_mha__bf16_head128_total32768/<timestamp-or-session-id>
```

Export it as `REMOTE_KDA_DIR`. Keep build outputs, profiler traces, NCU reports,
captured tensors, and benchmark logs inside that directory.

Before and after every benchmark or profile run, check GPU state and record the
selected host, GPU id, and GPU model in `benchmark.csv` or `docs/draft.md`.

## Workload Cases

Benchmark all combinations below:

| batch | seqlen | causal |
|---:|---:|---|
| 8 | 4096 | false |
| 8 | 4096 | true |
| 4 | 8192 | false |
| 4 | 8192 | true |
| 2 | 16384 | false |
| 2 | 16384 | true |
| 1 | 32768 | false |
| 1 | 32768 | true |

The final score may use the fastest correct variant per configured case, but
every dispatched variant must pass correctness for its assigned case.

## Implementation-Source Policy

This run is baseline-aware kernel evolution, not blind kernel synthesis.

Treat official FlashAttention-4, CUTLASS/CuTe SM100 examples, TileLang kernels,
and other public Blackwell attention kernels as reference and porting materials.
They may be studied or used as sources for license-compatible CUDA/C++,
CUTLASS/CuTe ports, and canonical helper code. Record exact source paths,
commits or installed versions, licenses when relevant, and what was adapted.

The final candidate implementation must be a native CUDA kernel built from
workspace-owned C++/CUDA source, such as `.cu`, `.cuh`, `.cpp`, or `.h` files
compiled with nvcc or an equivalent CUDA extension build. Python is allowed for
harnesses, bindings, benchmark scripts, and dispatch glue, but not as the
primary kernel implementation.

Do not use official FlashAttention-4, `flash_attn.cute.flash_attn_func`,
`FlashAttentionForwardSm100`, Python `cute.compile` over the FA4 CuTe DSL
kernel class, TileLang, Triton, torch SDPA, or any other prebuilt attention op
as the candidate execution path. These sources may only be inspected or ported
into native C++/CUDA/CUTLASS/CuTe code owned by this workspace.

The first performance-oriented candidate should be baseline-derived or
canonical-helper-derived unless there is a measured reason not to. A naive
kernel is acceptable only as a harness/correctness smoke test, not as the main
optimization lineage.

Do not hand-derive `tcgen05` SmemDescriptor encodings, TMEM layouts, TMA
swizzles, warpgroup synchronization protocols, or Blackwell MMA instruction
wrappers when an official or de facto canonical helper exists. Prefer porting
the helper and validating it with a microcase.

## Workspace Rule

All work for this kernel must happen inside the current prompt folder, the
folder that contains this `prompt.md`.

Required local files:

- `src/`: optimized wrapper, native CUDA/C++ sources, CUTLASS/CuTe helpers,
  build glue, dispatcher code, and registration entrypoint.
- `tests/`: isolated correctness tests and benchmark harness checks.
- `docs/draft.md`: implementation-plan draft written before code changes.
- `benchmark.csv`: every measured baseline and candidate comparison.
- `solutions.jsonl`: every candidate implementation, parent link, status, and
  evidence pointer.
- `profile/`: profiler traces and summaries.
- `ncu/`: Nsight Compute reports when collected.

## Baseline Recovery And Correctness Harness

Reference computation:

- Match standard scaled dot-product attention forward semantics for BF16 Q, K,
  and V with `head_dim=128`.
- Apply causal masking only when `causal=true`.
- Use a numerically stable online softmax/LSE-compatible formulation in the
  kernel.

Treat PyTorch/FP32 attention as the semantic correctness oracle.
FlashAttention-4 is the performance baseline and a useful cross-check, but it
is also a tiled BF16 implementation with its own reduction order.

Do not use a fixed `5e-3` absolute difference against FA4 as a hard correctness
gate for all cases. Use an SGLang-style dynamic numerical bound when practical:
compare the candidate error against the PyTorch/FP32 oracle to the error of a
PyTorch BF16 or reordered BF16 reference, and require the candidate to stay
within a small multiple of that numerical-error scale while also passing
NaN/Inf checks.

If the harness cannot cheaply compute a dynamic bound, keep the semantic
pass/fail gate on the PyTorch/FP32 oracle and use FA4 comparison as diagnostic
evidence. A relaxed FA4 cross-check such as `abs <= 2e-2` and `rel <= 0.10` may
catch gross divergence, but record it as methodology rather than the semantic
oracle.

Fill `tests/test_correctness.py` before optimization. The final candidate must
pass correctness before benchmark claims count.

## Benchmark Requirements

- Follow Dao-AILab/flash-attention `benchmarks/benchmark_attn.py` methodology as
  closely as practical, including warmup and repeat logic.
- Establish immutable FlashAttention-4 baseline numbers from the same selected
  idle B200 GPU and container before optimization claims.
- Report per-case mean latency, std, TFLOPS, and geometric mean TFLOPS.
- Keep benchmark scripts and raw result logs in this folder.
- Do not change the FA4 baseline, benchmark formula, warmup/repeat policy, or
  target cases after the first baseline is recorded unless the user explicitly
  asks for a methodology change. If a benchmark bug is found, record the
  before/after methodology in `benchmark.csv` and `docs/draft.md`.
- Use Nsight Compute when a correct candidate is not clearly target-complete or
  when profiler evidence would change the next edit.

## Prior Art Research Scope

Before choosing an implementation strategy, inspect SGLang, FlashAttention-4,
CUTLASS/CuTe, vLLM, TensorRT-LLM, FlashInfer, TileLang, Triton, PyTorch, and
relevant NVIDIA sample/plugin code. Use KernelWiki when prior B200, SM100,
FlashAttention-4, CUTLASS, CuTe, or attention-kernel evidence can guide a
design choice.

Record reviewed source paths and which ideas were kept or rejected in
`docs/draft.md` and `solutions.jsonl`.

## Optimization Exploration Policy

Use the KDA-style exploration loop:

- list candidate optimization directions in `docs/draft.md`;
- rank them by expected benefit, implementation risk, and how directly they
  attack the current measured bottleneck;
- try each direction for a bounded number of focused iterations;
- keep, revise, or reject each direction with correctness, benchmark, and NCU
  evidence;
- maintain parent links in `solutions.jsonl` so later runs can reconstruct the
  search DAG.

If a correct candidate is more than `3x` slower than official FA4 after one
tensor-core-capable attempt, stop local micro-tuning of that lineage and reset
to a stronger native CUDA/CUTLASS/CuTe porting parent.

If a tensor-core/TMEM/`tcgen05` microcase remains incorrect for two focused
iterations, stop hand-deriving that path and switch to canonical helper
extraction or a different parent implementation.

Consider B200/SM100-specific features and attention patterns such as TMA, TMEM,
`tcgen05` tensor-core MMA choices, warp specialization, persistent scheduling,
split-Q or split-K scheduling, online softmax/LSE, causal masking efficiency,
vectorized BF16 memory traffic, and occupancy vs register-pressure tradeoffs.

## Shape Specialization Policy

Shape-specialized kernels, template/config variants, causal/non-causal paths,
and a dispatcher or autotune table are allowed when measured evidence shows
that different sequence lengths or causal modes need different CTA, warpgroup,
TMEM, or register-pressure tradeoffs.

Record the dispatcher decision table with per-case baseline, candidate,
latency, TFLOPS, and promote/reject reason. Do not force a single universal
kernel if evidence shows that different sequence lengths or causal modes need
different tradeoffs.

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

`optimized_wrapper` must preserve the recovered MHA harness contract and fall
back to the baseline implementation for unsupported shapes, dtypes, layouts,
devices, head dimensions, head counts, or causal modes. See `interface.md`.

## Required Workflow

1. Confirm the current directory is this kernel folder.
2. Read `../../external/KernelWiki/SKILL.md` and
   `../../external/ncu-report-skill/SKILL.md` from this kernel folder.
3. Establish the immutable FA4 baseline, correctness command, benchmark command,
   first candidate direction, major risks, and promotion evidence.
4. Write an implementation-plan draft to `docs/draft.md`.
5. Run official Humanize plan generation on that draft.
6. Start official Humanize RLCR from this kernel folder.
7. Do not implement kernels, run long benchmarks, or collect NCU evidence before
   RLCR is active.
8. Record every candidate in `solutions.jsonl` and every performance result in
   `benchmark.csv`.

## Completion Bar

The work is complete only when:

- correctness tests pass for all configured cases;
- every dispatched variant is correct for its assigned case;
- B200 benchmark evidence shows at least `5%` geometric-mean TFLOPS improvement
  over official FlashAttention-4 across all configured cases, or a
  well-supported no-go conclusion explains why no defensible path remains under
  the available workspace;
- NCU evidence explains the improvement, blocker, and active hardware bound;
- `prompt.md`, `interface.md`, `benchmark.csv`, and `solutions.jsonl` are
  updated with the final result.
