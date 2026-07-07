# KDA Prompt: glm_52__sglang_unified_attention_with_output

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`
- `flashinfer.prefill.trtllm_ragged_attention_deepseek`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_standard_mha`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_trtllm`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_decode`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_extend`

## Environment And Runner

Use the `ion-b200` remote GPU environment for all B200 work. All CUDA, Python,
pip, nvcc, build, test, benchmark, and profiling commands must run inside the
existing `sglang_bbuf` Docker container on `ion-b200`.

Before GPU work, inspect `nvidia-smi` and choose a B200 GPU with no active
compute processes and no meaningful memory occupancy. Export that id as
`REMOTE_GPU_ID` and use it consistently for baseline, candidate, benchmark,
profiler, and NCU commands in the current run. Do not run measurements on busy
cards or directly on the `ion-b200` host.

**14.2% of total serving GPU time** on `zai-org/GLM-5.2-FP8` (cookbook-aligned profile, peak
`sharegpt_mid`) — a serving-profile headroom signal used to select this standalone kernel task. Family
`attention`, category `attention`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real GLM-5.2-FP8 TP=8 SGLang capture. Do not
start/re-run SGLang serve, `run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop;
optimize and validate via the task-local standalone benchmark on one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
