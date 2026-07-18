# Profile evidence - gpt_oss_120b__sglang_unified_attention_with_output

**Standalone kernel target: 8.4% of total serving GPU time** (max across scenarios) on
`openai/gpt-oss-120b`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `openai/gpt-oss-120b` production-path capture and replace the old noisy profiler shape strings.

- Model: `openai/gpt-oss-120b` (slug `gpt_oss_120b`, tp=8)
- Python interface(s): `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_decode`, `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend`
- Kernel family: `attention`  .  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16H64PagedKvCausalP64MultiCtasKvVarSeqQ8Kv128StaticSwap`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H64PagedKvCausalP64VarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H64PagedKvSlidingOrChunkedCausalP64VarSeqQ8Kv128Persi`, `void tensorrt_llm::kernels::quantize_with_block_size<(tensorrt_llm::BlockScaleQuantization`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 8.34% |
| random | conc 32 | 2.46% |
| sharegpt | conc 1 | 8.44% |
| sharegpt | conc 32 | 2.29% |
| sharegpt | conc 100 | 2.55% |

**Peak: 8.4% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 9
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real openai/gpt-oss-120b TP=8 SGLang production-path execution in temporary container sglang-gpt-oss-120b on eight B300 GPUs. Used a model-local HF cache after snapshot download, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy gpt_oss_120b__fp8_bmm task is routed to the real low-level FlashInfer TRTLLM batch context/decode APIs observed in the production path, while the attention task uses the higher-level TRTLLM backend forward APIs.

Functions covered:
- `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_decode`
- `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
