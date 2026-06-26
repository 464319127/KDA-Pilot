# Profile evidence — mimo_v25__fp8_bmm

**e2e-optimization target: 27.9% of total GPU time** (max across scenarios) on
`XiaomiMiMo/MiMo-V2.5`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `XiaomiMiMo/MiMo-V2.5` (slug `mimo_v25`, tp=4)
- Python interface: `<confirm via capture; profiler family=fp8_bmm>`
- Kernel family: `fp8_bmm`  ·  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_TN_transOut_noShfl`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128u2_s6_et64x64_m64x64x32_c1x1x1_rM_TN_transOut_noShfl`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128_s8_et64x8_m64x8x32_c1x1x1_rM_TN_transOut_noShfl_dsFp`, `bmm_E4m3_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_TN_transOut_noShfl_dsF`, `bmm_E4m3_E4m3E4m3_Fp32_t128x64x128u2_s6_et64x64_m64x64x32_c1x1x1_rM_TN_transOut_noShfl_dsF`, `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_TN_transOut_noShfl_dsFp8_`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.31% |
| random | conc 32 | 26.98% |
| random | conc 100 | 23.38% |
| sharegpt | conc 1 | 7.47% |
| sharegpt | conc 32 | 27.88% |
| sharegpt | conc 100 | 19.48% |

**Peak: 27.9% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[10096, 16, 192], [5745, 128, 2, 192], [5745, 128, 2, 128], [], [32], [], [], [`
- `[[10096, 16, 192], [7182, 128, 1, 192], [7182, 128, 1, 128], [], [32], [], [], [`
- `[[103, 256], [256], [103, 4096], [32, 103], [256, 1024, 4096], [256, 8, 32], [25`
- `[[10724, 256], [256], [10724, 4096], [32, 10724], [256, 1024, 4096], [256, 8, 32`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1], []]`
- `[[1]]`
- `[[2049, 1048580], []]`
- `[[256, 4096]]`
- `[[31], [], []]`
- `[[32, 89], [], [], []]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path XiaomiMiMo/MiMo-V2.5 --tp 4 --trust-remote-code --attention-backend fa4 --mm-attention-backend fa4 --moe-runner-backend flashinfer_trtllm --mem-fraction-static 0.65 --chunked-prefill-size 16384
```
After optimizing, re-run **sharegpt_mid** to validate the e2e effect.
