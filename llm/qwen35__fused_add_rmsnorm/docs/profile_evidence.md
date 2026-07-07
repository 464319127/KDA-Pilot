# Profile evidence — qwen35__fused_add_rmsnorm

**Standalone kernel target: 3.3% of total serving GPU time** (max across scenarios) on
`nvidia/Qwen3.5-397B-A17B-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/Qwen3.5-397B-A17B-NVFP4` (slug `qwen35`, tp=4)
- Python interface: `<confirm via capture; profiler family=fused_add_rmsnorm>`
- Kernel family: `fused_add_rmsnorm`  ·  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 2.52% |
| random | conc 100 | 3.31% |

**Peak: 3.3% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[1], [1], []]`
- `[[1]]`
- `[[38, 4096], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path nvidia/Qwen3.5-397B-A17B-NVFP4 --tp 4 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
