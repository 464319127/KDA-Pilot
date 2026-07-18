# Profile evidence - gpt_oss_120b__fp8_bmm

**Standalone kernel target: 15.8% of total serving GPU time** (max across scenarios) on
`openai/gpt-oss-120b`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `openai/gpt-oss-120b` production-path capture and replace the old noisy profiler shape strings.

- Model: `openai/gpt-oss-120b` (slug `gpt_oss_120b`, tp=8)
- Python interface(s): `flashinfer.decode.trtllm_batch_decode_with_kv_cache`, `flashinfer.prefill.trtllm_batch_context_with_kv_cache`
- Kernel family: `fp8_bmm`  .  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x128x128_s7_et128x64_m256x128x32_c2x1x1_rM_TN`, `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x128x256u2_s4_et128x64_m256x128x32_c2x1x1_rM_`, `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x16x256_s4_et128x16_m128x16x32_c1x1x1_rM_TN_t`, `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x256_s4_et128x8_m128x8x32_c1x1x1_rM_TN_tran`, `bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x256u2_s4_et128x8_m128x8x32_c1x1x1_rM_TN_tr`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x128x256_s4x4x4x4x1x4_et128x32_m256x128x32`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x128x256u2_s4_et128x32_m256x128x32_c2x1x1_`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x128x256u2_s4x4x4x4x1x4_et128x32_m256x128x`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x16x256_s5_et128x16_m256x16x32_c2x1x1_rM_T`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x16x256u2_s6_et128x16_m256x16x32_c2x1x1_rM`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512_s3_et128x8_m128x8x32_c1x1x1_rM_TN_t`, `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x512u2_s3_et128x8_m128x8x32_c1x1x1_rM_TN`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.18% |
| random | conc 32 | 15.81% |
| random | conc 100 | 14.03% |
| sharegpt | conc 1 | 5.23% |
| sharegpt | conc 32 | 15.29% |
| sharegpt | conc 100 | 8.82% |

**Peak: 15.8% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 18
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real openai/gpt-oss-120b TP=8 SGLang production-path execution in temporary container sglang-gpt-oss-120b on eight B300 GPUs. Used a model-local HF cache after snapshot download, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy gpt_oss_120b__fp8_bmm task is routed to the real low-level FlashInfer TRTLLM batch context/decode APIs observed in the production path, while the attention task uses the higher-level TRTLLM backend forward APIs.

Functions covered:
- `flashinfer.decode.trtllm_batch_decode_with_kv_cache`
- `flashinfer.prefill.trtllm_batch_context_with_kv_cache`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
