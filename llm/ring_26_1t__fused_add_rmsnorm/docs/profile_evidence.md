# Profile evidence — ring_26_1t__fused_add_rmsnorm

**Standalone kernel target: 28.9% of total serving GPU time** (max across scenarios) on
`inclusionAI/Ring-2.6-1T`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `inclusionAI/Ring-2.6-1T` (slug `ring_26_1t`, tp=8)
- Python interface: `<confirm via capture; profiler family=fused_add_rmsnorm>`
- Kernel family: `fused_add_rmsnorm`  ·  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 28.72% |
| random | conc 32 | 12.07% |
| random | conc 100 | 7.13% |
| sharegpt | conc 1 | 28.90% |
| sharegpt | conc 32 | 10.23% |
| sharegpt | conc 100 | 8.41% |

**Peak: 28.9% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[39, 256], [], [], [], [], [], []]`
- `[[39, 8192], [], []]`
- `[[44, 8192], [], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path inclusionAI/Ring-2.6-1T --tp-size 8 --trust-remote-code --mem-fraction-static 0.8 --tool-call-parser glm --reasoning-parser deepseek-r1
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
