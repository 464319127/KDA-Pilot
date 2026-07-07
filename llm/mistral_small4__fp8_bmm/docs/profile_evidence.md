# Profile evidence — mistral_small4__fp8_bmm

**Standalone kernel target: 42.3% of total serving GPU time** (max across scenarios) on
`mistralai/Mistral-Small-4-119B-2603`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `mistralai/Mistral-Small-4-119B-2603` (slug `mistral_small4`, tp=1)
- Python interface: `<confirm via capture; profiler family=fp8_bmm>`
- Kernel family: `fp8_bmm`  ·  Category: `other`
- GPU kernel(s): `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schPd2x`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schedS_`, `bmm_E4m3_E4m3E4m3_Fp32_BtokBfloat16_t128x8x256_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_`, `bmm_E4m3_E4m3E4m3_Fp32_BtokBfloat16_t128x8x256u2_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOu`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 14.47% |
| random | conc 32 | 29.44% |
| random | conc 100 | 42.29% |
| sharegpt | conc 1 | 15.10% |
| sharegpt | conc 32 | 35.91% |

**Peak: 42.3% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[1, 131072], [], []]`
- `[[100, 131072], [], []]`
- `[[100], [100], []]`
- `[[100]]`
- `[[1], [1], []]`
- `[[1], [], [], []]`
- `[[1], [], []]`
- `[[1]]`
- `[[256], [256], []]`
- `[[256], [], [], []]`
- `[[32], [32], []]`
- `[[32]]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path mistralai/Mistral-Small-4-119B-2603 --tp 1 --reasoning-parser mistral --tool-call-parser mistral
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
