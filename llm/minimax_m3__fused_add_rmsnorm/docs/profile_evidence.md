# Profile evidence — minimax_m3__fused_add_rmsnorm

**Standalone kernel target: 3.7% of total serving GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M3-MXFP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `MiniMaxAI/MiniMax-M3-MXFP8` (slug `minimax_m3`, tp=8)
- Python interface: `<confirm via capture; profiler family=fused_add_rmsnorm>`
- Kernel family: `fused_add_rmsnorm`  ·  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 3.57% |
| random | conc 100 | 3.21% |
| sharegpt | conc 1 | 3.72% |

**Peak: 3.7% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[1048580], [], [], []]`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1]]`
- `[[38], [], [], []]`
- `[[[128], [14]], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path MiniMaxAI/MiniMax-M3-MXFP8 --tp 8 --trust-remote-code --quantization mxfp8 --attention-backend fa4 --page-size 128 --mem-fraction-static 0.65
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
