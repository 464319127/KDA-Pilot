# Profile evidence — ring_26_1t__sgl_kernel_sgl_per_token_quant_fp8

**Standalone kernel target: 4.2% of total serving GPU time** (max across scenarios) on
`inclusionAI/Ring-2.6-1T`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `inclusionAI/Ring-2.6-1T` (slug `ring_26_1t`, tp=8)
- Python interface: `sgl_kernel.sgl_per_token_quant_fp8`
- Kernel family: `quant_fp8`  ·  Category: `quant_gemm`
- GPU kernel(s): `void per_token_quant_fp8_kernel<__nv_bfloat16, __nv_fp8_e4m3, 8, 16, false>(__nv_bfloat16 `, `void per_token_quant_fp8_small_batch_kernel<__nv_bfloat16, __nv_fp8_e4m3, 16>(__nv_bfloat1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 3.96% |
| random | conc 32 | 2.68% |
| random | conc 100 | 4.19% |
| sharegpt | conc 1 | 3.90% |
| sharegpt | conc 32 | 3.15% |
| sharegpt | conc 100 | 3.66% |

**Peak: 4.2% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[1, 1], [1, 1], []]`
- `[[1, 1], []]`
- `[[1, 2], [], [], []]`
- `[[14375, 256], [], [], [], [], [], []]`
- `[[14375, 8, 128], [14375, 8, 128], []]`
- `[[16384, 1, 64], []]`
- `[[16384, 8192], [16384, 8192], [16384, 1]]`
- `[[16384, 8192], [], [], [], [], [], []]`
- `[[1]]`
- `[[9780, 8192], [9780, 8192], [9780, 1]]`
- `[[], [131072, 512], [131072, 256]]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path inclusionAI/Ring-2.6-1T --tp-size 8 --trust-remote-code --mem-fraction-static 0.8 --tool-call-parser glm --reasoning-parser deepseek-r1
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
