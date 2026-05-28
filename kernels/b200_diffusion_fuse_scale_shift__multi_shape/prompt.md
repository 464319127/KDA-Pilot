# KDA Prompt: b200_diffusion_fuse_scale_shift__multi_shape

Optimize the SGLang diffusion `Fused scale_shift + dual-modulation (Z-Image adaLN)` kernel(s) for the full set of
production diffusion-model shapes captured from the SGLang diffusion benchmark
skill on NVIDIA B200. This prompt follows the Kernel Design Agents workflow
with official Humanize, KernelWiki, and `ncu-report-skill` available.

This is a multi-shape KDA prompt in the style of
`BBuf/kernel-design-agents`: first recover the baseline and correctness
contract, then run profiling-guided optimization, then specialize per shape
bucket with a dispatcher when measured evidence justifies it.

## Kernel Information

- Kernel folder: `b200_diffusion_fuse_scale_shift__multi_shape`
- Source project: SGLang (`python/sglang/jit_kernel/diffusion/`)
- Description: Optimize the SGLang diffusion fused-modulation Triton kernels: `fuse_scale_shift_kernel` (x*(1+scale)+shift on 3D BLC inputs with optional 4D scale/shift), `fuse_layernorm_scale_shift_gate_select01_kernel` (LN + adaLN + gate, dual-modulation per token), and `fuse_residual_layernorm_scale_shift_gate_select01_kernel` (additional residual+gate input).
- Hardware target: NVIDIA B200
- Wrapped baseline entry points:
- `sglang.jit_kernel.diffusion.triton.scale_shift:fuse_scale_shift_kernel`
- `sglang.jit_kernel.diffusion.triton.scale_shift:fuse_layernorm_scale_shift_gate_select01_kernel`
- `sglang.jit_kernel.diffusion.triton.scale_shift:fuse_residual_layernorm_scale_shift_gate_select01_kernel`
- Correctness oracle reference test:
`python/sglang/jit_kernel/tests/diffusion/test_qwen_image_modulation.py`
- Promotion target: at least 1.5x median-latency
speedup over the current SGLang baseline, computed as the geometric mean
of per-shape speedups across the configured shape table below.

## Workload Cases (Production Shapes)

These shapes were captured from the SGLang diffusion benchmark skill running
on the NVIDIA B200 reference host, plus derived from the upstream model
configurations. Every shape in the table below is part of the optimization
target.

| Preset | Model | dtype | x shape (B,L,C) | scale/shift layout | dual-modulation | residual | notes |
|---|---|---|---|---|---|---|---|
| flux | FLUX.1-dev | bfloat16 | (1, 4608, 3072) | (B,C) | No | No | adaLN modulation in joint attention |
| flux2 | FLUX.2-dev | bfloat16 | (1, 4608, 3072) | (B,C) | No | No | flux2 adaLN |
| qwen | Qwen-Image-2512 | bfloat16 | (1, 4352, 3072) | (B,C) | Yes (select01) | No | dual-modulation select |
| qwen-edit | Qwen-Image-Edit-2511 | bfloat16 | (1, 4608, 3072) | (B,C) | Yes (select01) | No | dual-modulation with edit |
| zimage | Z-Image-Turbo | bfloat16 | (1, 4096, 3072) | (B,C) | Yes (select01) | Yes | residual+gate+dual-modulation |
| wan-ti2v | Wan2.2-TI2V-5B | bfloat16 | (1, 75600, 3072) | (B,F,1,C) | No | No | 4D scale/shift over frames |
| wan-t2v | Wan2.2-T2V-A14B | bfloat16 | (1, 75600, 5120) | (B,F,1,C) | No | No | A14B frames modulation |
| wan-i2v | Wan2.2-I2V-A14B | bfloat16 | (1, 75600, 5120) | (B,F,1,C) | No | No | A14B I2V frames modulation |
| ltx2 | LTX-2 | bfloat16 | (1, 65520, 2048) | (B,C) | No | No | LTX-2 modulation |
| hunyuanvideo | HunyuanVideo | bfloat16 | (1, 33280, 3072) | (B,F,1,C) | No | No | Hunyuan frames modulation |
| mova-720p | MOVA-720p | bfloat16 | (1, 65536, 3072) | (B,F,1,C) | No | No | MOVA frames |
| helios | Helios-Base | bfloat16 | (1, 8448, 2048) | (B,C) | No | No | small video |


Shape collection methodology: the SGLang diffusion benchmark skill at
`~/.codex/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py`
was run for each preset with the `kernel_shape_capture.py` monkey-patch
active on `ion-b200` (B200) and `ion8-h200` / `ion9-h200` (H200). For
this kernel family live captures fired on presets `['qwen']` and are saved verbatim under `docs/captured_shapes_b200.jsonl` and
summarized in `docs/captured_shapes_b200.md` (3 unique
shape signatures). The analytical table above is the superset; any
additional shape observed in a future capture must be appended before
being claimed as part of the promotion target. Note: tensor shapes are
arch-independent for this kernel; if `captured_shapes_b200.jsonl` is empty
the agent must treat the H200 capture as the authoritative shape ledger.

## Canonical Regression Shapes (from SGLang test)

Source: `python/sglang/jit_kernel/tests/diffusion/test_qwen_image_modulation.py` (covers all three
scale-shift Triton entry points: `fuse_scale_shift_kernel`, the `select01` variant and the
residual `select01` variant).

- `batch_size`: `[1, 2, 4]` (CI subset `[1, 2]`).
- `seq_len`: `[6, 33, 128, 257]` (CI subset `[6, 128]`).
- `hidden_size`: `[512, 1024, 1536, 3072]` (CI subset `[512, 3072]`).
- `dtype`: `[float16, bfloat16, float32]` (CI subset `[fp16, bf16]`).
- `eps`: `1e-6`.
- `scale0/shift0/gate0/scale1/shift1/gate1`: each shaped `(B, C)`.
- `index`: shaped `(B, L)`, bool / int — required by the `select01` dispatch path.
- Residual / residual_gate path tested on the same grid via
  `fuse_residual_layernorm_scale_shift_gate_select01_kernel`.
- Tolerance: `(atol=5e-2, rtol=5e-2)` non-fp32; `1e-5` fp32.

The simple `fuse_scale_shift_kernel` accepts both 2D `(B,C)` scale/shift and 4D `(B,F,1,C)` scale/shift
(video frames). Cover both layouts during regression.

## Configurable Optimization Axes

Each candidate kernel/family may need different code paths or autotune
configs for different points in this axis space:

- x shape: B in [1,2] * L in [4096..75600] * C in [2048..5120]
- scale/shift layout: (B,C) / (1,C) / (B,F,1,C) / scalar
- dual-modulation index path (Qwen-Image-Edit / Z-Image)
- residual+gate path (Z-Image residual block)
- BLOCK_L / BLOCK_C / num_warps / num_stages choice
- dtype (bfloat16 / float16 / float32)

The promotion target is per-shape correctness with at least
1.5x geometric-mean speedup over the SGLang baseline
across **all** configured shapes. Per-shape specialization is allowed and
encouraged when profiler or benchmark evidence shows that one kernel cannot
cover the whole axis space. See the Shape Specialization Policy below.

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
/home/sglang-omni/bbuf/kda_runs/b200_diffusion_fuse_scale_shift__multi_shape/<timestamp-or-session-id>
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
`python/sglang/jit_kernel/tests/diffusion/test_qwen_image_modulation.py`.
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
wrapped entry point in `sglang/jit_kernel/diffusion/triton/scale_shift.py`.
2. Build a reproducible baseline harness for every shape in the shape table.
3. Adapt the SGLang reference correctness test `python/sglang/jit_kernel/tests/diffusion/test_qwen_image_modulation.py` into
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
KERNEL_SLUG = "b200_diffusion_fuse_scale_shift__multi_shape"
OP_TYPE = "fuse_scale_shift"

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
- NVIDIA B200 benchmark evidence shows at least 1.5x
geometric-mean median-latency speedup over the SGLang baseline across all
configured shape buckets, or a well-supported no-go conclusion explains
why no defensible path remains under the available workspace;
- NCU evidence explains the improvement, blocker, and active hardware
bound;
- `prompt.md`, `interface.md`, `benchmark.csv`, and `solutions.jsonl` are
updated with the final result.
