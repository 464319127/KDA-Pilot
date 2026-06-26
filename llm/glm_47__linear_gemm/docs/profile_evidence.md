# Profile evidence — glm_47__linear_gemm

**e2e-optimization target: 16.5% of total GPU time** (max across scenarios) on
`nvidia/GLM-4.7-NVFP4`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/GLM-4.7-NVFP4` (slug `glm_47`, tp=8)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `cutlass3x_sm100_tensorop_s128x64x8tf32gemm_f32_f32_f32_f32_f32_128x64x32_0_tnn_align4_2sm_`, `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPer`, `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_128x256_64x6_2x2_2cta_h_bz_bias_TNT`, `nvjet_sm100_tst_128x256_64x6_2x4_2cta_h_bz_bias_TNT`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_bias_TNN`, `nvjet_sm100_tst_64x32_64x16_2x4_2cta_h_bz_bias_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_splitK_bias_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 16.32% |
| random | conc 32 | 9.05% |
| random | conc 100 | 9.97% |
| sharegpt | conc 1 | 16.48% |
| sharegpt | conc 32 | 9.67% |
| sharegpt | conc 100 | 8.63% |

**Peak: 16.5% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[0], [0], []]`
- `[[1, 151552], [], [], []]`
- `[[1, 151552], [], []]`
- `[[100], [], []]`
- `[[128], [128], []]`
- `[[135], [], [], []]`
- `[[14], [], [], []]`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1], []]`
- `[[2432], [], [], []]`
- `[[2458], [], [], []]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model nvidia/GLM-4.7-NVFP4 --tp 8 --quantization modelopt_fp4 --reasoning-parser glm45 --trust-remote-code
```
After optimizing, re-run **sharegpt_low** to validate the e2e effect.
