# Profile evidence — glm_52__sglang_unified_attention_with_output

**Standalone kernel target: 14.2% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-5.2-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below were recaptured from a real GLM-5.2-FP8 TP=8 server run and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-5.2-FP8` (slug `glm_52`, tp=8)
- Python interface(s): `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`, `flashinfer.prefill.trtllm_ragged_attention_deepseek`, `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_standard_mha`, `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_trtllm`, `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_decode`, `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_extend`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1Mult`, `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1VarSeqQ8Kv128Pe`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.76% |
| random | conc 100 | 4.33% |
| sharegpt | conc 1 | 3.28% |
| sharegpt | conc 32 | 14.21% |
| sharegpt | conc 100 | 4.60% |

**Peak: 14.2% in `sharegpt_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 46
- Capture note: Captured on verda-b300-fin-03-1/light-face-hides-fin-03-1, sglang-glm52 container, zai-org/GLM-5.2-FP8 tp=8, CUDA graphs disabled for Python API capture, 2026-07-07. Final records include DSA/TRTLLM attention, torch.bmm, DeepGEMM, and per-token quant call contracts from real server requests.

Functions covered:
- `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`
- `flashinfer.prefill.trtllm_ragged_attention_deepseek`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_standard_mha`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_trtllm`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_decode`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_extend`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path zai-org/GLM-5.2-FP8 --tp 8 --trust-remote-code --mem-fraction-static 0.8
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
