# Profile evidence — ling_26__linear_gemm

**Standalone kernel target: 12.1% of total serving GPU time** (max across scenarios) on
`inclusionAI/Ling-2.6-flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `inclusionAI/Ling-2.6-flash` (slug `ling_26`, tp=4)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `gemm`
- GPU kernel(s): `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x128x16_1x1x1_3_tnn_align1_bias_f32_relu`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu`, `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_256x224_64x4_2x2_2cta_h_bz_TNT`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x8_64x16_4x2_h_bz_TNT`, `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.55% |
| random | conc 32 | 12.12% |
| random | conc 100 | 4.19% |
| sharegpt | conc 1 | 2.38% |
| sharegpt | conc 32 | 11.98% |
| sharegpt | conc 100 | 2.55% |

**Peak: 12.1% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[100], [100], []]`
- `[[107], [107], []]`
- `[[11]]`
- `[[12199, 4096], [4096, 256]]`
- `[[12199, 4096], [4096, 4608]]`
- `[[1], [1], []]`
- `[[1]]`
- `[[2247], [], [], []]`
- `[[32], [32], []]`
- `[[380], [], [], [], []]`
- `[[384], [], [], []]`
- `[[39, 4096], [4096, 256]]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path inclusionAI/Ling-2.6-flash --tp 4 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
