# Profile evidence - glm_47__sglang_unified_attention_with_output

**Standalone kernel target: 4.9% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-4.7-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `nvidia/GLM-4.7-NVFP4` production-path capture and replace the old noisy profiler shape strings.

- Model: `nvidia/GLM-4.7-NVFP4` (slug `glm_47`, tp=8)
- Python interface(s): `flashinfer.decode.trtllm_batch_decode_with_kv_cache`, `flashinfer.prefill.trtllm_batch_context_with_kv_cache`, `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_decode`, `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend`
- Kernel family: `attention`  .  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvE4m3OBfloat16H128PagedKvCausalP64MultiCtasKvVarSeqQ16Kv128StaticSwapsA`, `fmhaSm100fKernel_QkvE4m3OBfloat16H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 4.64% |
| random | conc 100 | 3.69% |
| sharegpt | conc 1 | 4.87% |
| sharegpt | conc 32 | 3.32% |
| sharegpt | conc 100 | 4.34% |

**Peak: 4.9% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 18
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real nvidia/GLM-4.7-NVFP4 TP=8 SGLang production-path execution in temporary container sglang-glm47. Used a model-local HF cache in offline mode after snapshot download, trust_remote_code, quantization=modelopt_fp4, reasoning_parser=glm45, attention_backend=trtllm_mha/FlashInfer default for this NVFP4 checkpoint, moe_runner_backend=flashinfer_trtllm as selected by SGLang for ModelOpt NVFP4 MoE, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy fp8_bmm task maps to the captured FlashInfer NVFP4 mm_fp4 Python API because the old profiler class name did not correspond to a bmm_fp8 Python call in this production path.

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
