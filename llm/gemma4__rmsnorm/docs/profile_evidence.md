# Profile evidence — gemma4__rmsnorm

**Standalone kernel target: 18.9% of total serving GPU time** (max across scenarios) on
`google/gemma-4-26B-A4B-it`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `google/gemma-4-26B-A4B-it` (slug `gemma4`, tp=1)
- Python interface: `<confirm via capture; profiler family=rmsnorm>`
- Kernel family: `rmsnorm`  ·  Category: `gemm`
- GPU kernel(s): `_gemma_dual_rmsnorm_residual_kernel`, `_gemma_qkv_rmsnorm_kernel`, `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 18.58% |
| random | conc 32 | 9.54% |
| random | conc 100 | 7.03% |
| sharegpt | conc 1 | 18.89% |
| sharegpt | conc 32 | 9.75% |
| sharegpt | conc 100 | 7.46% |

**Peak: 18.9% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[1, 1], [1, 1], []]`
- `[[1, 262144], [], []]`
- `[[17, 8192], [], [], []]`
- `[[1785, 16, 256], [1785, 8, 256], [262400, 256], [1785], [], []]`
- `[[1785, 8, 256], []]`
- `[[1902, 2816], [2816, 128]]`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1], []]`
- `[[1]]`
- `[[2049, 262148], []]`
- `[[38, 2048], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path google/gemma-4-26B-A4B-it --reasoning-parser gemma4 --tool-call-parser gemma4
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
