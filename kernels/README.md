# KernelPilot Kernel Tasks

KernelPilot now follows the Kernel Design Agents release pattern: each task is a
self-contained kernel prompt folder, and implementation work happens inside an
isolated worktree launched from `scripts/launch_kernels/`.

Each kernel folder contains:

```text
prompt.md              # source task prompt
interface.md           # expected exported candidate interface
benchmark.py           # isolated baseline-vs-candidate timing scaffold
benchmark.csv          # append-only benchmark evidence ledger
solutions.jsonl        # append-only candidate lineage ledger
docs/                  # plan drafts, source notes, run logs
profile/               # torch-profiler traces and summaries
ncu/                   # Nsight Compute reports
src/                   # optimized implementation and wrapper code
tests/test_correctness.py
```

Use the matching launcher under `../scripts/launch_kernels/` to start a task.

## SGLang Diffusion Multi-Shape Tasks

The `*_diffusion_multi_shape/` folders cover SGLang's non-gemm / non-attention
diffusion kernels under `python/sglang/jit_kernel/diffusion/`. Their shape
tables come from a live sweep of the SGLang diffusion benchmark presets with
a `kernel_shape_capture.py` monkey-patch active. See:

- `diffusion_shapes_ledger.md` — cross-task summary of every observed shape.
- `diffusion_kernel_coverage.md` — kernel × preset coverage matrix, including
  which entries are empirical vs analytical fallbacks.
- `<task>/docs/captured_shapes_<arch>.{jsonl,md}` — raw and rendered captures
  for one task.
- `../scripts/diffusion_shape_capture/` — the capture and replay tooling.
