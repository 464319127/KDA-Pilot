# Profile evidence — glm_5__fp8_bmm

**Standalone kernel target: 19.5% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-5-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/GLM-5-NVFP4` (slug `glm_5`, tp=4)
- Python interface: `<confirm via capture; profiler family=fp8_bmm>`
- Kernel family: `fp8_bmm`  ·  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x128x256u2_s6_et128x128_m256x128x64_c2x1x1_rM_TN_`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x16x512u2_s5_et128x16_m128x16x64_c1x1x1_rM_TN_tr`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOu`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512u2_s5_et128x8_m128x8x64_c1x1x1_rM_TN_trans`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.50% |
| random | conc 32 | 19.50% |
| random | conc 100 | 17.70% |
| sharegpt | conc 1 | 5.25% |
| sharegpt | conc 32 | 14.88% |
| sharegpt | conc 100 | 11.56% |

**Peak: 19.5% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[1, 1], [1, 1], []]`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[201]]`
- `[[20784, 6144], [6144, 128]]`
- `[[20784, 6144], [6144, 32], [], [20784, 32]]`
- `[[2128, 202756], [], [32], [], []]`
- `[[30796, 1, 64], [30796, 1, 64], []]`
- `[[30796, 16, 256], [], []]`
- `[[30796, 16, 64], [30796, 1, 64], [203008, 64], [30796], [], []]`
- `[[31], [31], []]`
- `[[31], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path nvidia/GLM-5-NVFP4 --tp 4 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
