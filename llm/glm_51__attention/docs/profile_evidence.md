# Profile evidence - glm_51__attention

**Standalone kernel target: 3.8% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-5.1-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `zai-org/GLM-5.1-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-5.1-FP8` (slug `glm_51`, tp=8)
- Python interface(s): `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`, `flashinfer.prefill.trtllm_ragged_attention_deepseek`, `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_standard_mha`, `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_trtllm`, `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_decode`, `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_extend`
- Kernel family: `attention`  .  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1VarSeqQ8Kv128Pe`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| sharegpt | conc 32 | 3.82% |

**Peak: 3.8% in `sharegpt_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 58
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real zai-org/GLM-5.1-FP8 TP=8 SGLang production-path execution in temporary container sglang-glm51. Used a model-local HF cache after snapshot download, quantization=fp8 as detected by SGLang, kv_cache_dtype=fp8_e4m3, attention_backend=dsa with TRTLLM DSA prefill/decode, moe_runner_backend=flashinfer_trtllm, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The generic attention task maps to the captured DSA backend forward APIs because this GLM-5.1 route uses DSA/TRTLLM MLA attention rather than a standalone generic attention Python API.

Functions covered:
- `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`
- `flashinfer.prefill.trtllm_ragged_attention_deepseek`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_standard_mha`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_trtllm`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_decode`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_extend`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
