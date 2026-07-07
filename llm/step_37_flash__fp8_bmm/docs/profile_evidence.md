# Profile evidence — step_37_flash__fp8_bmm

**Standalone kernel target: 3.6% of total serving GPU time** (max across scenarios) on
`stepfun-ai/Step-3.7-Flash-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `stepfun-ai/Step-3.7-Flash-NVFP4` (slug `step_37_flash`, tp=8)
- Python interface: `<confirm via capture; profiler family=fp8_bmm>`
- Kernel family: `fp8_bmm`  ·  Category: `quant_gemm`
- GPU kernel(s): `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x16x512u2_s5_et128x16_m128x16x64_c1x1x1_rM_TN_tr`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 100 | 3.57% |

**Peak: 3.6% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[100], [100], []]`
- `[[1]]`
- `[[223], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path stepfun-ai/Step-3.7-Flash-NVFP4 --tp 8 --ep 8 --moe-runner-backend flashinfer_trtllm --kv-cache-dtype fp8_e4m3
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
