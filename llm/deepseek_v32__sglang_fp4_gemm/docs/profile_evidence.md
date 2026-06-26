# Profile evidence — deepseek_v32__sglang_fp4_gemm

**e2e-optimization target: 32.5% of total GPU time** (max across scenarios) on
`nvidia/DeepSeek-V3.2-NVFP4`, from the exact cookbook-aligned profile. Clean Python interface (profiler provenance).

- Model: `nvidia/DeepSeek-V3.2-NVFP4` (slug `deepseek_v32`, tp=4)
- Python interface: `sglang.fp4_gemm`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPer`, `nvjet_sm100_tst_128x256_64x4_1x2_h_bz_TNT`, `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x16_64x16_2x4_2cta_h_bz_splitK_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_splitK_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`, `void flashinfer::trtllm_dsv3_router_gemm::router_gemm_kernel<__nv_bfloat16, float, 128, 8,`, `void fused_a_gemm_kernel<1, 2112, 7168, 16, 8, 256, 16>(__nv_bfloat16*, __nv_bfloat16 cons`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 31.74% |
| random | conc 32 | 10.28% |
| random | conc 100 | 9.24% |
| sharegpt | conc 1 | 32.50% |
| sharegpt | conc 32 | 12.82% |
| sharegpt | conc 100 | 16.24% |

**Peak: 32.5% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[0], [], []]`
- `[[1, 103], [], [], []]`
- `[[1, 129280], [], []]`
- `[[1, 12], [1, 12], []]`
- `[[1, 2], [1, 2], []]`
- `[[1, 725], []]`
- `[[103, 4], []]`
- `[[103], [], [], []]`
- `[[13285, 1536], [1536, 8192]]`
- `[[13285, 3584], [3584, 9216], [13312, 448], [448, 9216], [], [], []]`
- `[[13285, 7168], [7168, 2112]]`
- `[[13285, 7168], [7168, 256]]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model nvidia/DeepSeek-V3.2-NVFP4 --tp 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3
```
After optimizing, re-run **sharegpt_low** to validate the e2e effect.
