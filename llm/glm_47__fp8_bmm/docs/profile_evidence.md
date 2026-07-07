# Profile evidence — glm_47__fp8_bmm

**Standalone kernel target: 19.5% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-4.7-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/GLM-4.7-NVFP4` (slug `glm_47`, tp=8)
- Python interface: `<confirm via capture; profiler family=fp8_bmm>`
- Kernel family: `fp8_bmm`  ·  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_tr`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x32x256_s9_et128x32_m128x32x64_c1x1x1_rM_TN_trans`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_r`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x16x512_s5_et128x16_m256x16x64_c2x1x1_rM_T`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x16x512_s5_et128x16_m256x16x64_c2x1x1_rM_TN_tran`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x16x512u2_s5_et128x16_m128x16x64_c1x1x1_rM_TN_tr`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x32x512_s4_et128x32_m128x32x64_c1x1x1_rM_TN_tran`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x32x512u2_s4_et128x32_m128x32x64_c1x1x1_rM_TN_tr`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512u2_s5_et128x8_m128x8x64_c1x1x1_rM_TN_trans`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 7.35% |
| random | conc 32 | 12.48% |
| random | conc 100 | 19.49% |
| sharegpt | conc 1 | 5.34% |
| sharegpt | conc 32 | 12.80% |
| sharegpt | conc 100 | 18.73% |

**Peak: 19.5% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[100], [], []]`
- `[[100], []]`
- `[[1024, 1, 128], [], [], []]`
- `[[1024, 1536], [], [], []]`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1], []]`
- `[[1]]`
- `[[32], [], []]`
- `[[32], []]`
- `[[332], [], [], []]`
- `[[384], [], [], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model nvidia/GLM-4.7-NVFP4 --tp 8 --quantization modelopt_fp4 --reasoning-parser glm45 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
