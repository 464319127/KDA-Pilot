# Profile evidence — glm_5__fused_add_rmsnorm

**Standalone kernel target: 4.1% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-5-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/GLM-5-NVFP4` (slug `glm_5`, tp=4)
- Python interface: `<confirm via capture; profiler family=fused_add_rmsnorm>`
- Kernel family: `fused_add_rmsnorm`  ·  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 3.20% |
| random | conc 100 | 4.13% |
| sharegpt | conc 32 | 2.03% |
| sharegpt | conc 100 | 3.03% |

**Peak: 4.1% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[16873, 2048], [2048, 4096]]`
- `[[20784, 20784], [20784], [20784, 2048], [60, 2896], [61], [20784]]`
- `[[9962, 2048], [2048, 4096]]`
- `[[9962, 32, 1], []]`
- `[[9962, 6144], [], [], [], [], [], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path nvidia/GLM-5-NVFP4 --tp 4 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
