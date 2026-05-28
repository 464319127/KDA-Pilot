# KDA Prompt: h200_diffusion_qknorm_rope__multi_shape

Optimize the SGLang diffusion `Fused QKNorm + RoPE (CUDA, in-place)` kernel(s) for the full set of
production diffusion-model shapes captured from the SGLang diffusion benchmark
skill on NVIDIA H200. This prompt follows the Kernel Design Agents workflow
with official Humanize, KernelWiki, and `ncu-report-skill` available.

This is a multi-shape KDA prompt in the style of
`BBuf/kernel-design-agents`: first recover the baseline and correctness
contract, then run profiling-guided optimization, then specialize per shape
bucket with a dispatcher when measured evidence justifies it.

## Kernel Information

- Kernel folder: `h200_diffusion_qknorm_rope__multi_shape`
- Source project: SGLang (`python/sglang/jit_kernel/diffusion/`)
- Description: Fuse per-head RMS normalization on Q and K with RoPE position rotation in a single in-place CUDA kernel. Baseline is the current SGLang `fused_inplace_qknorm_rope` CUDA implementation (templated by head_dim, rope_dim, is_neox, dtype).
- Hardware target: NVIDIA H200
- Wrapped baseline entry points:
- `sglang.jit_kernel.diffusion.qknorm_rope:fused_inplace_qknorm_rope`
- Correctness oracle reference test:
`python/sglang/jit_kernel/tests/diffusion/test_qknorm_rope.py`
- Promotion target: at least 1.3x median-latency
speedup over the current SGLang baseline, computed as the geometric mean
of per-shape speedups across the configured shape table below.

The baseline is a native CUDA kernel (templated by head_dim, rope_dim, is_neox, dtype) loaded via JIT. The optimized candidate must also be a native C++/CUDA kernel built from workspace-owned `.cu`/`.cuh`/`.cpp`/`.h` sources, not pure Python.

## Workload Cases (Production Shapes)

These shapes were captured from the SGLang diffusion benchmark skill running
on the NVIDIA H200 reference host, plus derived from the upstream model
configurations. Every shape in the table below is part of the optimization
target.

| Preset | Model | dtype | total_tokens (B*S) | num_heads_q | num_heads_k | head_dim | rope_dim | is_neox | notes |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| flux | FLUX.1-dev | bfloat16 | 4608 | 24 | 24 | 128 | 128 | False | 1024^2 @ patch=2; 4096 image + 512 text tokens |
| flux2 | FLUX.2-dev | bfloat16 | 4608 | 24 | 24 | 128 | 128 | False | 1024^2 @ patch=2; ~4608 joint tokens |
| qwen | Qwen-Image-2512 | bfloat16 | 4352 | 24 | 24 | 128 | 128 | True | 1024^2 transformer joint qkv |
| qwen-edit | Qwen-Image-Edit-2511 | bfloat16 | 4608 | 24 | 24 | 128 | 128 | True | image + edit conditioning |
| zimage | Z-Image-Turbo | bfloat16 | 4096 | 24 | 24 | 128 | 128 | False | residual-form modulation pipeline |
| wan-ti2v | Wan2.2-TI2V-5B | bfloat16 | 75600 | 24 | 24 | 128 | 128 | False | 720p, 81 frames, patch=(1,2,2) |
| wan-t2v | Wan2.2-T2V-A14B | bfloat16 | 75600 | 40 | 40 | 128 | 128 | False | 720p, 81 frames, A14B branch |
| wan-i2v | Wan2.2-I2V-A14B | bfloat16 | 75600 | 40 | 40 | 128 | 128 | False | 720p, 81 frames, image conditioning |
| ltx2 | LTX-2 | bfloat16 | 65520 | 24 | 24 | 128 | 96 | False | 1536x1024, 121 frames, split rotary, rope_dim<head_dim |
| hunyuanvideo | HunyuanVideo | bfloat16 | 33280 | 24 | 24 | 128 | 128 | False | 848x480, 65 frames |
| mova-720p | MOVA-720p | bfloat16 | 65536 | 24 | 24 | 128 | 128 | False | 720p talking-face, 193 frames |
| helios | Helios-Base | bfloat16 | 8448 | 16 | 16 | 128 | 128 | False | 640x384, 33 frames |


Shape collection methodology: the SGLang diffusion benchmark skill at
`~/.codex/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py`
was run for each preset with the `kernel_shape_capture.py` monkey-patch
active on `ion-b200` (B200) and `ion8-h200` / `ion9-h200` (H200). For
this kernel family live captures fired on presets `['qwen', 'zimage']` and are saved verbatim under `docs/captured_shapes_h200.jsonl` and
summarized in `docs/captured_shapes_h200.md` (6 unique
shape signatures). The analytical table above is the superset; any
additional shape observed in a future capture must be appended before
being claimed as part of the promotion target. Note: tensor shapes are
arch-independent for this kernel; if `captured_shapes_b200.jsonl` is empty
the agent must treat the H200 capture as the authoritative shape ledger.

## Canonical Regression Shapes (from SGLang test)

Source: `python/sglang/jit_kernel/tests/diffusion/test_qknorm_rope.py`.
Every candidate must still pass this enumerated grid in nightly /
`base-b-kernel-unit-1-gpu-large` CI:

- `batch_size` (== total tokens before view): every power of two in `[1, 4096]`
  plus `x+1` for each, plus CI-extra `[1, 9, 129, 257, 2049, 4097]`.
- `num_heads`: `[8, 16, 24, 32]` (CI subset `[8, 24]`).
- `head_dim`: `[64, 128, 256]`.
- `rope_dim`: `{64: [64], 128: [64, 128], 256: [64, 128, 256]}` per `head_dim`.
- `is_neox`: `[False, True]`; when `True`, only `rope_dim` values yielding a power-of-two
  rotary-lane count are valid (see `can_use_fused_inplace_qknorm_rope` gate).
- `position_dtype`: `[torch.int32, torch.int64]`.
- `dtype`: `torch.bfloat16`; `eps=1e-6`; tolerance `ATOL=8e-2, RTOL=1e-2`.
- Oracle: SGLang `fused_inplace_qknorm` + `flashinfer.rope.apply_rope_with_cos_sin_cache_inplace`.

The candidate kernel must support every `(batch_size, num_heads, head_dim, rope_dim, is_neox, position_dtype)`
tuple above or fall back to the SGLang baseline for the unsupported tail.

## Configurable Optimization Axes

Each candidate kernel/family may need different code paths or autotune
configs for different points in this axis space:

- head_dim (64 / 128 / 256)
- rope_dim (64 / 96 / 128, <= head_dim)
- is_neox (True / False)
- dtype (bfloat16 / float16)
- total_tokens range (4096 - 75600)
- num_heads (16 / 24 / 32 / 40)

The promotion target is per-shape correctness with at least
1.3x geometric-mean speedup over the SGLang baseline
across **all** configured shapes. Per-shape specialization is allowed and
encouraged when profiler or benchmark evidence shows that one kernel cannot
cover the whole axis space. See the Shape Specialization Policy below.

## Environment And Remote Rule

Use the `ion8-h200 (or ion9-h200 as backup)` remote GPU environment for all NVIDIA H200
work. All CUDA, Python, pip, nvcc, build, test, benchmark, and Nsight Compute
commands must run inside the existing `sglang_bbuf` Docker container on
`ion8-h200`, with an idle NVIDIA H200 GPU selected.

Before running GPU work, inspect `nvidia-smi` and choose a GPU with no active
compute processes and no meaningful memory occupancy. Export that id as
`REMOTE_GPU_ID` and use it consistently for the baseline, candidate,
benchmark, profiler, and NCU commands in the current run.

Use this command pattern for remote execution:

```bash
ssh ion8-h200 'REMOTE_GPU_ID=<idle-gpu-id>; docker exec sglang_bbuf bash -lc "CUDA_VISIBLE_DEVICES=${REMOTE_GPU_ID} <command>"'
```

Do not run Python, pip, nvcc, builds, tests, benchmarks, or profiling
directly on the `ion8-h200` host.

When multiple sessions share the same remote container, create a task-owned
remote workspace under:

```text
/home/sglang-omni/bbuf/kda_runs/h200_diffusion_qknorm_rope__multi_shape/<timestamp-or-session-id>
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
`python/sglang/jit_kernel/tests/diffusion/test_qknorm_rope.py`.
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
wrapped entry point in `sglang/jit_kernel/diffusion/qknorm_rope.py`.
2. Build a reproducible baseline harness for every shape in the shape table.
3. Adapt the SGLang reference correctness test `python/sglang/jit_kernel/tests/diffusion/test_qknorm_rope.py` into
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
idle NVIDIA H200 GPU and container environment.
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
KernelWiki when prior SM90, CUTLASS, SGLang, or normalization /
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

Consider SM90-specific optimization paths:

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
KERNEL_SLUG = "h200_diffusion_qknorm_rope__multi_shape"
OP_TYPE = "qknorm_rope_inplace"

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
- NVIDIA H200 benchmark evidence shows at least 1.3x
geometric-mean median-latency speedup over the SGLang baseline across all
configured shape buckets, or a well-supported no-go conclusion explains
why no defensible path remains under the available workspace;
- NCU evidence explains the improvement, blocker, and active hardware
bound;
- `prompt.md`, `interface.md`, `benchmark.csv`, and `solutions.jsonl` are
updated with the final result.
