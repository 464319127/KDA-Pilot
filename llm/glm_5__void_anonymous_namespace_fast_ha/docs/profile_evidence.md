# Profile evidence - glm_5__void_anonymous_namespace_fast_ha

**Standalone kernel target: 3.7% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-5-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `nvidia/GLM-5-NVFP4` production-path capture and replace the old noisy profiler shape strings.

- Model: `nvidia/GLM-5-NVFP4` (slug `glm_5`, tp=4)
- Python interface(s): `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_standard_mha`, `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_trtllm`, `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_decode`, `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_extend`
- Kernel family: `void_anonymous_namespace_fast_ha`  .  Category: `other`
- GPU kernel(s): `void (anonymous namespace)::fast_hadamard_transform_kernel<(anonymous namespace)::FastHada`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| sharegpt | conc 32 | 2.73% |
| sharegpt | conc 100 | 3.73% |

**Peak: 3.7% in `sharegpt_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 22
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real nvidia/GLM-5-NVFP4 TP=4 SGLang production-path execution in temporary container sglang-glm5. Used a model-local HF cache after snapshot download, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, attention_backend=dsa with TRTLLM DSA prefill/decode as selected by SGLang, moe_runner_backend=flashinfer_trtllm, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, and marked five request windows covering long prefill, short decode, mid concurrency, high concurrency, and an extra-long DSA/indexer prompt. The legacy fast_hadamard task maps to the captured DSA backend forward APIs because the current cookbook path did not emit a standalone hadamard_transform Python API; the old fast_hadamard GPU work is fused inside the DSA path.

Functions covered:
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
