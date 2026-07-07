# Profile evidence — step_37_flash__linear_gemm

**Standalone kernel target: 12.9% of total serving GPU time** (max across scenarios) on
`stepfun-ai/Step-3.7-Flash-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `stepfun-ai/Step-3.7-Flash-NVFP4` (slug `step_37_flash`, tp=8)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `gemm`
- GPU kernel(s): `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x128x16_1x1x1_3_tnn_align1_bias_f32_relu`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x128x16_1x1x1_3_tnn_align1_bias_f32_relu`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu`, `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.60% |
| random | conc 32 | 12.93% |
| random | conc 100 | 6.88% |
| sharegpt | conc 1 | 2.58% |
| sharegpt | conc 32 | 4.48% |

**Peak: 12.9% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[12438, 4096], [4096, 2816]]`
- `[[12438, 4096], [4096, 288]]`
- `[[1], [1], []]`
- `[[1]]`
- `[[256], [256], []]`
- `[[2977, 4096], [4096, 288]]`
- `[[4096, 160], [], []]`
- `[[576], [576], []]`
- `[[64], []]`
- `[[704], [], [], []]`
- `[[8510, 4096], [4096, 288]]`
- `[[858], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path stepfun-ai/Step-3.7-Flash-NVFP4 --tp 8 --ep 8 --moe-runner-backend flashinfer_trtllm --kv-cache-dtype fp8_e4m3
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
