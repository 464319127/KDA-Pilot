# Profile evidence — gpt_oss_120b__fp8_bmm

**Standalone kernel target: 15.8% of total serving GPU time** (max across scenarios) on
`openai/gpt-oss-120b`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `openai/gpt-oss-120b` (slug `gpt_oss_120b`, tp=8)
- Python interface: `<confirm via capture; profiler family=fp8_bmm>`
- Kernel family: `fp8_bmm`  ·  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x128x128_s7_et128x64_m256x128x32_c2x1x1_rM_TN`, `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x128x256u2_s4_et128x64_m256x128x32_c2x1x1_rM_`, `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x16x256_s4_et128x16_m128x16x32_c1x1x1_rM_TN_t`, `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x256_s4_et128x8_m128x8x32_c1x1x1_rM_TN_tran`, `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x256u2_s4_et128x8_m128x8x32_c1x1x1_rM_TN_tr`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x128x256_s4x4x4x4x1x4_et128x32_m256x128x32`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x128x256u2_s4_et128x32_m256x128x32_c2x1x1_`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x128x256u2_s4x4x4x4x1x4_et128x32_m256x128x`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x16x256_s5_et128x16_m256x16x32_c2x1x1_rM_T`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x16x256u2_s6_et128x16_m256x16x32_c2x1x1_rM`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512_s3_et128x8_m128x8x32_c1x1x1_rM_TN_t`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512u2_s3_et128x8_m128x8x32_c1x1x1_rM_TN`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.18% |
| random | conc 32 | 15.81% |
| random | conc 100 | 14.03% |
| sharegpt | conc 1 | 5.23% |
| sharegpt | conc 32 | 15.29% |
| sharegpt | conc 100 | 8.82% |

**Peak: 15.8% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[1, 201088], [], [], []]`
- `[[1, 201088], [], []]`
- `[[159], [], [], []]`
- `[[160, 512], [], [], [], []]`
- `[[18]]`
- `[[1], [1], []]`
- `[[1]]`
- `[[256], [256], []]`
- `[[261], [], [], []]`
- `[[32]]`
- `[[384], [384], []]`
- `[[4097], [], [13], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path openai/gpt-oss-120b --tp 8
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
