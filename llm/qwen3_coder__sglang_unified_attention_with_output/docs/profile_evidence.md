# Profile evidence - qwen3_coder__sglang_unified_attention_with_output

**Standalone kernel target: 6.8% of total serving GPU time** (max across scenarios) on
`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` (slug `qwen3_coder`, tp=8)
- Python interface(s): `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_decode`, `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend`
- Kernel family: `attention`  .  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP32MultiCtasKvCgaVarSeqQ16Kv128Stati`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP32VarSeqQ16Kv128PersistentSwapsAbFo`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 4.68% |
| random | conc 100 | 3.18% |
| sharegpt | conc 1 | 4.71% |
| sharegpt | conc 32 | 2.67% |
| sharegpt | conc 100 | 6.80% |

**Peak: 6.8% in `sharegpt_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 9
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 TP=8 EP=8 SGLang production-path execution in temporary container sglang-qwen3-coder on eight B300 GPUs. Used a model-local HF cache after snapshot download, tool_call_parser=qwen3_coder, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy qwen3_coder__fp8_bmm task is routed to the real low-level FlashInfer TRTLLM batch context/decode APIs observed in the production path, while the attention task uses the higher-level TRTLLM backend forward APIs.

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
