# KDA Prompt: qwen36__fused_add_rmsnorm

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.srt.layers.layernorm.gemma_fused_add_rmsnorm`

**3.5% of total serving GPU time** on `Qwen/Qwen3.6-35B-A3B-FP8` (cookbook-aligned
profile, peak `sharegpt_low`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `fused_add_rmsnorm`, category `gemm`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `Qwen/Qwen3.6-35B-A3B-FP8` TP=1 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
