# KDA Prompt: glm_5__void_anonymous_namespace_fast_ha

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_standard_mha`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._forward_trtllm`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_decode`
- `sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.forward_extend`

**3.7% of total serving GPU time** on `nvidia/GLM-5-NVFP4` (cookbook-aligned
profile, peak `sharegpt_high`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `void_anonymous_namespace_fast_ha`, category `other`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `nvidia/GLM-5-NVFP4` TP=4 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
