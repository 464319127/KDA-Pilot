# Profile evidence - qwen36__fp8_bmm

**Standalone kernel target: 15.9% of total serving GPU time** (max across scenarios) on
`Qwen/Qwen3.6-35B-A3B-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `Qwen/Qwen3.6-35B-A3B-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `Qwen/Qwen3.6-35B-A3B-FP8` (slug `qwen36`, tp=1)
- Python interface(s): `flashinfer.decode.trtllm_batch_decode_with_kv_cache`, `flashinfer.prefill.trtllm_batch_context_with_kv_cache`, `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_decode`, `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend`
- Kernel family: `fp8_bmm`  .  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_E4m3E4m3_Fp32_t128x16x128_s6_et64x16_m64x16x32_c1x1x1_rM_TN_transOut_noShfl_d`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_TN_transOut_noShfl_ds`, `bmm_E4m3_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_TN_transOut_noShfl_dsF`, `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_TN_transOut_noShfl_dsFp8_`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 15.94% |
| random | conc 100 | 9.06% |
| sharegpt | conc 1 | 6.49% |
| sharegpt | conc 100 | 4.67% |

**Peak: 15.9% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 43
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real Qwen/Qwen3.6-35B-A3B-FP8 TP=1 SGLang production-path execution in temporary container sglang-qwen36 on a single B300 GPU. Used a model-local HF cache after snapshot download, speculative_algorithm=EAGLE with 3 steps/topk 1 per the cookbook command, quantization=fp8 as detected by SGLang, attention_backend=trtllm_mha, moe_runner_backend=flashinfer_trtllm, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy fp8_bmm task maps to captured TRTLLM MHA attention APIs because this Qwen3.6 route did not emit a standalone bmm Python API; the old bmm GPU work is fused inside the attention backend.

Functions covered:
- `flashinfer.decode.trtllm_batch_decode_with_kv_cache`
- `flashinfer.prefill.trtllm_batch_context_with_kv_cache`
- `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_decode`
- `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
