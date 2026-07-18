# Profile evidence - mimo_v25__fp8_bmm

**Standalone kernel target: 27.9% of total serving GPU time** (max across scenarios) on
`XiaomiMiMo/MiMo-V2.5`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `XiaomiMiMo/MiMo-V2.5` production-path capture and replace the old noisy profiler shape strings.

- Model: `XiaomiMiMo/MiMo-V2.5` (slug `mimo_v25`, tp=4)
- Python interface(s): `sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.forward_decode`, `sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.forward_extend`, `sglang.srt.layers.attention.flashattention_backend.flash_attn_with_kvcache`
- Kernel family: `fp8_bmm`  .  Category: `quant_gemm`
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

**Peak: 27.9% in `sharegpt_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 64
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real XiaomiMiMo/MiMo-V2.5 TP=4 SGLang production-path execution in temporary container sglang-mimo-v25 with FA4 attention and flashinfer_trtllm MoE; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; request windows include long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.forward_decode`
- `sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.forward_extend`
- `sglang.srt.layers.attention.flashattention_backend.flash_attn_with_kvcache`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
