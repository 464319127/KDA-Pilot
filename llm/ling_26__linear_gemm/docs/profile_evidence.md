# Profile evidence - ling_26__linear_gemm

**Standalone kernel target: 12.1% of total serving GPU time** (max across scenarios) on
`inclusionAI/Ling-2.6-flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `inclusionAI/Ling-2.6-flash` production-path capture and replace the old noisy profiler shape strings.

- Model: `inclusionAI/Ling-2.6-flash` (slug `ling_26`, tp=4)
- Python interface(s): `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `gemm`
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

**Peak: 12.1% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 19
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real inclusionAI/Ling-2.6-flash TP=4 SGLang production-path execution in temporary container sglang-ling26; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; request windows include long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
