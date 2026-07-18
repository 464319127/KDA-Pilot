# KDA Prompt: gpt_oss_120b__rmsnorm

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.srt.layers.layernorm.rmsnorm`

**7.0% of total serving GPU time** on `openai/gpt-oss-120b` (cookbook-aligned
profile, peak `random_mid`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `rmsnorm`, category `norm`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `openai/gpt-oss-120b` TP=8 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
