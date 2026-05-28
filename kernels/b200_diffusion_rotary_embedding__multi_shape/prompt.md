# KDA Prompt: b200_diffusion_rotary_embedding__multi_shape

Optimize the SGLang diffusion `Standard and LTX-2 split rotary embeddings` kernel(s) for the full set of
production diffusion-model shapes captured from the SGLang diffusion benchmark
skill on NVIDIA B200. This prompt follows the Kernel Design Agents workflow
with official Humanize, KernelWiki, and `ncu-report-skill` available.

This is a multi-shape KDA prompt in the style of
`BBuf/kernel-design-agents`: first recover the baseline and correctness
contract, then run profiling-guided optimization, then specialize per shape
bucket with a dispatcher when measured evidence justifies it.

## Kernel Information

- Kernel folder: `b200_diffusion_rotary_embedding__multi_shape`
- Source project: SGLang (`python/sglang/jit_kernel/diffusion/`)
- Description: Optimize the two SGLang diffusion RoPE kernels: `apply_rotary_embedding` (standard interleaved / non-interleaved RoPE used by most DiTs) and `apply_ltx2_split_rotary_emb` (LTX-2 split rotary that consumes a 4D `(B, H, S, head_dim/2)` cos/sin layout).
- Hardware target: NVIDIA B200
- Wrapped baseline entry points:
- `sglang.jit_kernel.diffusion.triton.rotary:apply_rotary_embedding`
- `sglang.jit_kernel.diffusion.triton.ltx2_rotary:apply_ltx2_split_rotary_emb`
- Correctness oracle reference test:
`python/sglang/jit_kernel/tests/test_rope.py`
- Promotion target: at least 1.4x median-latency
speedup over the current SGLang baseline, computed as the geometric mean
of per-shape speedups across the configured shape table below.

## Workload Cases (Production Shapes)

These shapes were captured from the SGLang diffusion benchmark skill running
on the NVIDIA B200 reference host, plus derived from the upstream model
configurations. Every shape in the table below is part of the optimization
target.

| Preset | Model | kernel | dtype | x layout | total tokens | num_heads | head_dim | interleaved | notes |
|---|---|---|---|---|---:|---:|---:|---|---|
| flux | FLUX.1-dev | apply_rotary_embedding | bfloat16 | (B*S, H, D) | 4608 | 24 | 128 | False | joint qk RoPE |
| flux2 | FLUX.2-dev | apply_rotary_embedding | bfloat16 | (B*S, H, D) | 4608 | 24 | 128 | False | joint qk RoPE |
| qwen | Qwen-Image-2512 | apply_rotary_embedding | bfloat16 | (B*S, H, D) | 4352 | 24 | 128 | True | NeoX RoPE |
| qwen-edit | Qwen-Image-Edit-2511 | apply_rotary_embedding | bfloat16 | (B*S, H, D) | 4608 | 24 | 128 | True | edit RoPE |
| zimage | Z-Image-Turbo | apply_rotary_embedding | bfloat16 | (B*S, H, D) | 4096 | 24 | 128 | False | Z-Image RoPE |
| wan-ti2v | Wan2.2-TI2V-5B | apply_rotary_embedding | bfloat16 | (B*S, H, D) | 75600 | 24 | 128 | False | 720p RoPE |
| wan-t2v | Wan2.2-T2V-A14B | apply_rotary_embedding | bfloat16 | (B*S, H, D) | 75600 | 40 | 128 | False | A14B RoPE |
| wan-i2v | Wan2.2-I2V-A14B | apply_rotary_embedding | bfloat16 | (B*S, H, D) | 75600 | 40 | 128 | False | A14B RoPE |
| ltx2 | LTX-2 | apply_ltx2_split_rotary_emb | bfloat16 | (B, S, inner_dim) | 65520 | 24 | 128 | n/a | rope_dim=96, split half_dim |
| hunyuanvideo | HunyuanVideo | apply_rotary_embedding | bfloat16 | (B*S, H, D) | 33280 | 24 | 128 | False | 848x480 RoPE |
| mova-720p | MOVA-720p | apply_rotary_embedding | bfloat16 | (B*S, H, D) | 65536 | 24 | 128 | False | 720p RoPE |
| helios | Helios-Base | apply_rotary_embedding | bfloat16 | (B*S, H, D) | 8448 | 16 | 128 | False | 640x384 RoPE |


Shape collection methodology: the SGLang diffusion benchmark skill at
`~/.codex/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py`
was run for each preset with the `kernel_shape_capture.py` monkey-patch
active on `ion-b200` (B200) and `ion8-h200` / `ion9-h200` (H200). For
this kernel family live captures fired on presets `['ltx2']` and are saved verbatim under `docs/captured_shapes_b200.jsonl` and
summarized in `docs/captured_shapes_b200.md` (4 unique
shape signatures). The analytical table above is the superset; any
additional shape observed in a future capture must be appended before
being claimed as part of the promotion target. Note: tensor shapes are
arch-independent for this kernel; if `captured_shapes_b200.jsonl` is empty
the agent must treat the H200 capture as the authoritative shape ledger.

## Canonical Regression Shapes (from SGLang test)

Source: `python/sglang/jit_kernel/tests/test_rope.py` (standard `apply_rotary_embedding` enumeration).

- `batch_size` (== total tokens): `[1, 2, 4, ..., 2048]` (powers of two) plus `[1, 129, 2048, 2049]` CI-extra.
- `num_kv_heads`: `[1, 2, 8]` (CI subset `[1, 8]`).
- `gqa_ratio`: `[1, 4, 8]` (CI subset `[1, 8]`). `num_qo_heads = num_kv_heads * gqa_ratio`.
- `rope_dim`: `[64, 128, 256, 512]` (CI subset `[64, 256]`).
- `is_neox`: `[False, True]`.
- `dtype`: bfloat16 default plus mixed `[int32, int64]` for the `positions` dtype edge case.
- Oracle: FlashInfer `apply_rope_with_cos_sin_cache_inplace`. Tolerance `1e-2` abs/rel.

For the LTX-2 split-rotary variant (`apply_ltx2_split_rotary_emb`) there is no dedicated SGLang test;
the empirical shapes captured from the `ltx2` benchmark preset are the regression contract.

## Configurable Optimization Axes

Each candidate kernel/family may need different code paths or autotune
configs for different points in this axis space:

- head_dim (64 / 96 / 128 / 256)
- num_heads (16 / 24 / 32 / 40)
- interleaved (True / False)
- x layout: 3D (S,H,D) vs 4D (B,S,H,D)
- cos/sin shape: (S, D) vs (S, D/2) vs (B, H, S, D/2)
- split rotary (LTX-2): half_dim != head_dim/2 path
- total tokens range (4096 - 75600)
- dtype (bfloat16 / float16)

The promotion target is per-shape correctness with at least
1.4x geometric-mean speedup over the SGLang baseline
across **all** configured shapes. Per-shape specialization is allowed and
encouraged when profiler or benchmark evidence shows that one kernel cannot
cover the whole axis space. See the Shape Specialization Policy below.

## Required Claude Code Skill

This task talks to the remote GPU box exclusively through the local
Claude Code skill `~/.claude/skills/ion-b200/SKILL.md`. The skill
owns the SSH alias `ion-b200`, the `sglang_bbuf` Docker container
lifecycle (the create command keeps `--privileged --cap-add=SYS_ADMIN
--security-opt seccomp=unconfined` so `ncu --set basic` can collect
counters), the idle-GPU selection rule (`nvidia-smi` 0% util + low
memory), and the `kill-idle` shortcut.

If the skill is missing on the box that launches Humanize/RLCR, fetch
it before starting the loop; do not paraphrase the SSH pattern by
hand. The skill's SKILL.md is the single source of truth for `ion-b200`
host conventions; this prompt only consumes them.

## Environment And Remote Rule

Use the `ion-b200` remote GPU environment for all NVIDIA B200
work. All CUDA, Python, pip, nvcc, build, test, benchmark, and Nsight Compute
commands must run inside the existing `sglang_bbuf` Docker container on
`ion-b200`, with an idle NVIDIA B200 GPU selected.

Before running GPU work, inspect `nvidia-smi` and choose a GPU with no active
compute processes and no meaningful memory occupancy. Export that id as
`REMOTE_GPU_ID` and use it consistently for the baseline, candidate,
benchmark, profiler, and NCU commands in the current run.

Use this command pattern for remote execution:

```bash
ssh ion-b200 'REMOTE_GPU_ID=<idle-gpu-id>; docker exec sglang_bbuf bash -lc "CUDA_VISIBLE_DEVICES=${REMOTE_GPU_ID} <command>"'
```

Do not run Python, pip, nvcc, builds, tests, benchmarks, or profiling
directly on the `ion-b200` host.

When multiple sessions share the same remote container, create a task-owned
remote workspace under:

```text
/home/sglang-omni/bbuf/kda_runs/b200_diffusion_rotary_embedding__multi_shape/<timestamp-or-session-id>
```

Export it as `REMOTE_KDA_DIR`. Keep build outputs, profiler traces, NCU
reports, captured tensors, and benchmark logs inside that directory.

Before and after every benchmark or profile run, check GPU state and record
the selected host, GPU id, and GPU model in `benchmark.csv` or
`docs/draft.md`.

## Workspace Rule

All work for this kernel task must happen inside the current prompt folder,
the folder that contains this `prompt.md`.

Required local files:

- `src/`: optimized wrapper, native sources (CUDA `.cu`/`.cuh`/`.cpp` for the
CUDA family, Triton/CuTe-DSL Python for the others), dispatcher code, and
registration entrypoint.
- `tests/`: correctness tests adapted from the SGLang reference test under
`python/sglang/jit_kernel/tests/test_rope.py`.
- `docs/draft.md`: implementation-plan draft written before code changes.
- `docs/shapes_<host>.jsonl`: captured shape JSONL from the diffusion
benchmark sweep, copied into this folder.
- `benchmark.csv`: every measured baseline vs candidate comparison.
- `solutions.jsonl`: every candidate implementation, parent link, status,
and evidence pointer.
- `profile/`: profiler traces and summaries.
- `ncu/`: Nsight Compute reports when collected.

Keep SGLang checkouts as dependencies to inspect, run, or patch
temporarily. The optimized implementation and evidence for this task stay
local to this folder unless intentionally promoted later.

## Baseline Recovery And Correctness Harness

Recover the baseline before writing optimized code:

1. Inspect the SGLang baseline implementation file(s) corresponding to each
wrapped entry point in `sglang/jit_kernel/diffusion/triton/rotary.py`.
2. Build a reproducible baseline harness for every shape in the shape table.
3. Adapt the SGLang reference correctness test `python/sglang/jit_kernel/tests/test_rope.py` into
`tests/test_correctness.py` of this folder, parameterized over the
configured shape buckets.
4. Capture or generate baseline inputs covering the relevant dtypes and
layouts above.
5. Preserve explicit NaN/Inf checks in every validator.
6. Use dynamic numerical tolerances where applicable
(SGLang-style: candidate error must not exceed a small multiple of the
reference BF16/FP16 quantization noise vs FP32).

The final candidate must pass correctness for every configured shape
before benchmark claims count.

## Benchmark Requirements

- Use warmup and repeated timing.
- Report median latency, mean latency, std, min, p10, and p90 per shape.
- Compare every candidate against the SGLang baseline from the same selected
idle NVIDIA B200 GPU and container environment.
- Keep benchmark scripts and raw result logs in this folder.
- Every claimed improvement must identify the candidate commit or file
version and the exact command used to produce the result.
- Use Nsight Compute when a correct candidate is not clearly target-complete
or when profiler evidence would change the next edit.
- Final claim must be the geometric mean of per-shape speedups across the
full shape table, not the best-case shape alone.

## Prior Art Research Scope

Before choosing an implementation strategy, inspect SGLang, CUTLASS/CuTe,
CUDA samples, PyTorch, vLLM, TensorRT-LLM, FlashInfer, DeepGEMM, and
public Blackwell or Hopper kernels for directly relevant ideas. Use
KernelWiki when prior SM100, CUTLASS, SGLang, or normalization /
modulation / RoPE / group-norm evidence can guide a design choice.

Record reviewed source paths, commits or installed versions, and which
ideas were kept or rejected in `docs/draft.md` and `solutions.jsonl`.

## Optimization Exploration Policy

Use the KDA-style exploration loop:

- list candidate optimization directions in `docs/draft.md`;
- rank them by expected benefit, implementation risk, and how directly
they attack the measured bottleneck;
- try each direction for a bounded number of focused iterations;
- keep, revise, or reject each direction with correctness, benchmark, and
NCU evidence;
- maintain parent links in `solutions.jsonl` so later runs can reconstruct
the search DAG.

Consider SM100-specific optimization paths:

- `tcgen05` instructions or warp-specialized cooperative MMA where the
kernel touches a small inner matmul,
- TMEM / TMA / cluster shapes when the working set exceeds shared memory,
- persistent or split scheduling along (rows, hidden, frames, groups),
- vectorized loads/stores (LDG.128 / STG.128) keyed to dtype and stride,
- shared-memory staging vs register-file pressure tradeoffs,
- fused-epilogue patterns when the kernel chains norm + scale-shift +
gate + residual.

## Shape Specialization Policy

Shape-specialized kernels, template/config variants, and a dispatcher or
autotune table are allowed when measured evidence shows that different
shape buckets need different CTA, warpgroup, TMEM, or register-pressure
tradeoffs.

Record the dispatcher decision table with per-bucket baseline, candidate,
latency, speedup, and promote/reject reason in `benchmark.csv` and a
`docs/dispatch.md` note. Do not force a single universal kernel if
evidence shows that different shape buckets need different tradeoffs.

The shape buckets to consider include but are not limited to:

- small image (B*S in 4096..6144, D in 2048..3072) - FLUX/Qwen-Image/Z-Image
- large video (B*S in 33000..75600, D in 2048..5120) - Wan2.2/HunyuanVideo/MOVA/LTX-2
- small video (B*S in 8000..16000, D in 2048..3072) - Helios
- high-channel low-token (Wan A14B with D=5120) vs low-channel high-token (LTX-2 D=2048)

## Interface Contract

Add the candidate under `src/` and expose:

```text
src/register.py
```

with:

```python
KERNEL_SLUG = "b200_diffusion_rotary_embedding__multi_shape"
OP_TYPE = "rotary_embedding"

def optimized_wrapper(*args, **kwargs):
...

def register() -> dict:
return {
"name": KERNEL_SLUG,
"op_type": OP_TYPE,
"callable": optimized_wrapper,
"version": "dev",
"source": __file__,
}
```

`optimized_wrapper` must preserve the recovered SGLang callsite contract and
fall back to the baseline implementation for unsupported shapes, dtypes,
layouts, devices, normalization types, or feature flags. See
`interface.md` for the exact signature contract to be recovered from the
baseline.

## Required Workflow

1. Confirm the current directory is this kernel folder.
2. Read `../../external/KernelWiki/SKILL.md` and
`../../external/ncu-report-skill/SKILL.md` from this kernel folder.
3. Recover the SGLang baseline path, tensor contract, and exact benchmark
command for every shape in the shape table.
4. Copy the captured shape JSONL into `docs/shapes_<host>.jsonl`.
5. Write an implementation-plan draft to `docs/draft.md`.
6. Run official Humanize plan generation on that draft.
7. Start official Humanize RLCR from this kernel folder.
8. Do not implement kernels, run long benchmarks, or collect NCU evidence
before RLCR is active.
9. Record every candidate in `solutions.jsonl` and every performance result
in `benchmark.csv`.

## Completion Bar

The work is complete only when:

- correctness tests pass for every configured shape;
- every dispatched variant is correct for its assigned shape bucket;
- NVIDIA B200 benchmark evidence shows at least 1.4x
geometric-mean median-latency speedup over the SGLang baseline across all
configured shape buckets, or a well-supported no-go conclusion explains
why no defensible path remains under the available workspace;
- NCU evidence explains the improvement, blocker, and active hardware
bound;
- `prompt.md`, `interface.md`, `benchmark.csv`, and `solutions.jsonl` are
updated with the final result.
