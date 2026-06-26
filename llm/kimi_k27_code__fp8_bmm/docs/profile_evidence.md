# Profile evidence — kimi_k27_code__fp8_bmm

**e2e-optimization target: 28.4% of total GPU time** (max across scenarios) on
`moonshotai/Kimi-K2.7-Code`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `moonshotai/Kimi-K2.7-Code` (slug `kimi_k27_code`, tp=8)
- Python interface: `<confirm via capture; profiler family=fp8_bmm>`
- Kernel family: `fp8_bmm`  ·  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_Ab32_t128x128x128_s4_et128x64_m256x128x16_c2`, `bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_Ab32_t128x16x256_s3_et128x16_m256x16x16_c2x1`, `bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_Ab32_t128x16x256u2_s3_et128x16_m256x16x16_c2`, `bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_Ab32_t128x32x256_s3_et128x32_m256x32x16_c2x1`, `bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_Ab32_t128x32x256u2_s3_et128x32_m256x32x16_c2`, `bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_Ab32_t128x64x256_s3_et128x64_m128x64x16_c1x1`, `bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_Ab32_t128x64x256u2_s3_et128x64_m128x64x16_c1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 17.13% |
| random | conc 100 | 16.93% |
| sharegpt | conc 1 | 2.00% |
| sharegpt | conc 32 | 28.39% |
| sharegpt | conc 100 | 10.65% |

**Peak: 28.4% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[1, 2], [], [], []]`
- `[[100], [100], []]`
- `[[104], [], [], []]`
- `[[15128], [], [], [], []]`
- `[[161], [], [], []]`
- `[[1]]`
- `[[2816, 7168]]`
- `[[3177, 1024], [1024, 7168]]`
- `[[3177, 7168], [7168, 512]]`
- `[[320], [320], []]`
- `[[32], [32], []]`
- `[[32], [], [], []]`

## Reproduce (cookbook-aligned)
```bash
sglang serve --model-path moonshotai/Kimi-K2.7-Code --tp 8 --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --trust-remote-code
```
After optimizing, re-run **sharegpt_mid** to validate the e2e effect.
