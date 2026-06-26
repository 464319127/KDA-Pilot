# Profile evidence — deepseek_v32__fp8_bmm

**e2e-optimization target: 21.2% of total GPU time** (max across scenarios) on
`nvidia/DeepSeek-V3.2-NVFP4`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/DeepSeek-V3.2-NVFP4` (slug `deepseek_v32`, tp=4)
- Python interface: `<confirm via capture; profiler family=fp8_bmm>`
- Kernel family: `fp8_bmm`  ·  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x128x256u2_s6_et128x128_m256x128x64_c2x1x1_rM_TN_`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x16x512_s5_et128x16_m256x16x64_c2x1x1_rM_TN_trans`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256u2_s6_et128x128_m256x128x64_c2x1x1`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x16x512_s5_et128x16_m256x16x64_c2x1x1_rM_T`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x16x512_s5_et128x16_m128x16x64_c1x1x1_rM_TN_tran`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x16x512u2_s5_et128x16_m128x16x64_c1x1x1_rM_TN_tr`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOu`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512u2_s5_et128x8_m128x8x64_c1x1x1_rM_TN_trans`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 8.65% |
| random | conc 32 | 11.33% |
| random | conc 100 | 21.16% |
| sharegpt | conc 1 | 4.39% |
| sharegpt | conc 32 | 12.68% |
| sharegpt | conc 100 | 17.65% |

**Peak: 21.2% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[1, 103], [], [], []]`
- `[[1, 12], [1, 12], []]`
- `[[1, 2], [1, 2], []]`
- `[[1, 730], []]`
- `[[100], [100], []]`
- `[[100], []]`
- `[[104, 129280], [], [], []]`
- `[[104, 16], [104, 16], []]`
- `[[13285, 6144], []]`
- `[[13285, 7168]]`
- `[[192], [], [], []]`
- `[[1], [1], []]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model nvidia/DeepSeek-V3.2-NVFP4 --tp 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3
```
After optimizing, re-run **random_high** to validate the e2e effect.
