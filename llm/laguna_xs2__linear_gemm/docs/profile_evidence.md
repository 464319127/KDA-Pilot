# Profile evidence — laguna_xs2__linear_gemm

**e2e-optimization target: 42.4% of total GPU time** (max across scenarios) on
`poolside/Laguna-XS.2-NVFP4`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `poolside/Laguna-XS.2-NVFP4` (slug `laguna_xs2`, tp=1)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalINS1_17GroupProblemShapeIN4cute5t`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_32x32x16_1x1x1_3_tnn_align1_bias_f32_relu`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_tnn_align1_bias_f32_relu`, `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPer`, `nvjet_sm100_tst_128x192_64x7_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_128x256_64x6_2x2_2cta_h_bz_TNT`, `nvjet_sm100_tst_256x128_64x5_2x4_2cta_h_bz_TNT`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_splitK_TNT`, `nvjet_sm100_tst_80x64_64x12_4x1_v_bz_TNN`, `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 31.60% |
| random | conc 32 | 42.07% |
| random | conc 100 | 39.88% |
| sharegpt | conc 1 | 30.96% |
| sharegpt | conc 32 | 36.11% |
| sharegpt | conc 100 | 42.43% |

**Peak: 42.4% in `sharegpt_high` (sharegpt, concurrency 100).**

## Input shapes (profiler)
- `[[1, 100352], [], [], []]`
- `[[1013057], []]`
- `[[132, 8192], [132, 8192], []]`
- `[[144, 8192], [144, 8, 128], [144, 8, 128], [144, 8192], [], [], [], [], [], [],`
- `[[1886, 8192], [1886, 8192], []]`
- `[[1], [1], []]`
- `[[1], [], [], []]`
- `[[1]]`
- `[[27], [27], []]`
- `[[27]]`
- `[[29], [29], []]`
- `[[32], [32], []]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path poolside/Laguna-XS.2-NVFP4 --tp 1 --trust-remote-code
```
After optimizing, re-run **sharegpt_high** to validate the e2e effect.
