# Profile evidence - qwen3__fp8_bmm

**Standalone kernel target: 32.0% of total serving GPU time** (max across scenarios) on
`Qwen/Qwen3-235B-A22B-Instruct-2507`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `Qwen/Qwen3-235B-A22B-Instruct-2507` production-path capture and replace the old noisy profiler shape strings.

- Model: `Qwen/Qwen3-235B-A22B-Instruct-2507` (slug `qwen3`, tp=8)
- Python interface(s): `flashinfer.decode.trtllm_batch_decode_with_kv_cache`, `flashinfer.prefill.trtllm_batch_context_with_kv_cache`
- Kernel family: `fp8_bmm`  .  Category: `other`
- GPU kernel(s): `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x128x64u2_s6_et128x128_m256x128x16_c2x1x1_rM_BN_tra`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x16x128_s6_et128x16_m128x16x16_c1x1x1_rM_BN_transOu`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x16x128u2_s3_et128x16_m128x16x16_c1x1x1_rM_BN_trans`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x16x128u2_s5_et128x16_m128x16x16_c1x1x1_rM_BN_trans`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x64x128_s5_et128x64_m256x64x16_c2x1x1_rM_BN_transOu`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x64x128u2_s5_et128x64_m256x64x16_c2x1x1_rM_BN_trans`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128_s4_et128x8_m128x8x16_c1x1x1_rM_BN_transOut_s`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128_s5_et128x8_m128x8x16_c1x1x1_rM_BN_transOut_s`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128u2_s4_et128x8_m128x8x16_c1x1x1_rM_BN_transOut`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128u2_s5_et128x8_m128x8x16_c1x1x1_rM_BN_transOut`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x8x128u2_s6_et128x8_m128x8x16_c1x1x1_rM_BN_transOut`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 16.27% |
| random | conc 32 | 31.99% |
| random | conc 100 | 23.82% |
| sharegpt | conc 1 | 10.81% |
| sharegpt | conc 32 | 27.31% |
| sharegpt | conc 100 | 24.30% |

**Peak: 32.0% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 9
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real Qwen/Qwen3-235B-A22B-Instruct-2507 TP=8 SGLang production-path execution in temporary container sglang-qwen3 on eight B300 GPUs. Used a model-local HF cache after snapshot download, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy qwen3__fp8_bmm task is routed to the real low-level FlashInfer TRTLLM batch context/decode APIs observed in the production path, while the attention task uses the higher-level TRTLLM backend forward APIs. The qwen3__void_cublas_lt_split_kreduce_ker profiler family has no separate Python API in this capture; its workload is duplicated from the real torch.nn.functional.linear API that emits the cublasLt splitK reduce GPU kernel.

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
