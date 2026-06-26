# Profile evidence — mimo_v25__linear_gemm

**e2e-optimization target: 9.2% of total GPU time** (max across scenarios) on
`XiaomiMiMo/MiMo-V2.5`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `XiaomiMiMo/MiMo-V2.5` (slug `mimo_v25`, tp=4)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `gemm`
- GPU kernel(s): `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_tnn_align1_bias_f32_relu`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu`, `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x`, `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 8.79% |
| random | conc 100 | 5.91% |
| sharegpt | conc 1 | 2.07% |
| sharegpt | conc 32 | 9.17% |
| sharegpt | conc 100 | 2.61% |

**Peak: 9.2% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[10096], []]`
- `[[103], []]`
- `[[10724, 4096], [4096, 256]]`
- `[[170, 4096], [4096, 256]]`
- `[[196, 4096], [4096, 256]]`
- `[[1], [], []]`
- `[[2049, 1048580], [], [], [], []]`
- `[[32], []]`
- `[[5674, 4096], [4096, 256]]`
- `[[89, 4096], [4096, 256]]`
- `[[]]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path XiaomiMiMo/MiMo-V2.5 --tp 4 --trust-remote-code --attention-backend fa4 --mm-attention-backend fa4 --moe-runner-backend flashinfer_trtllm --mem-fraction-static 0.65 --chunked-prefill-size 16384
```
After optimizing, re-run **sharegpt_mid** to validate the e2e effect.
