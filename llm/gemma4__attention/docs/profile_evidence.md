# Profile evidence - gemma4__attention

**Standalone kernel target: 12.0% of total serving GPU time** (max across scenarios) on
`google/gemma-4-26B-A4B-it`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `google/gemma-4-26B-A4B-it` production-path capture and replace the old noisy profiler shape strings.

- Model: `google/gemma-4-26B-A4B-it` (slug `gemma4`, tp=1)
- Python interface(s): `flashinfer.decode.trtllm_batch_decode_with_kv_cache`, `flashinfer.prefill.trtllm_batch_context_with_kv_cache`, `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_decode`, `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend`
- Kernel family: `attention`  .  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvSlidingOrChunkedCausalP64MultiCtasKvCgaVar`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvSlidingOrChunkedCausalP64VarSeqQ128Kv128Pe`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvSlidingOrChunkedCausalP64VarSeqQ8Kv128Pers`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 4.07% |
| random | conc 32 | 5.69% |
| random | conc 100 | 12.04% |
| sharegpt | conc 1 | 4.14% |
| sharegpt | conc 32 | 6.32% |
| sharegpt | conc 100 | 7.92% |

**Peak: 12.0% in `random_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 36
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real google/gemma-4-26B-A4B-it TP=1 SGLang production-path execution in temporary container sglang-gemma4 on one B300 GPU. Used a model-local HF cache after snapshot download, cookbook-aligned reasoning_parser=gemma4 and tool_call_parser=gemma4, disabled CUDA graph prefill/decode, delayed torch-op wrapping until after imports, preserved server startup/warmup records, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

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
