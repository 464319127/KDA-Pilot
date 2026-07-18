# KDA Prompt: step35_flash__void_moe_top_k_256_float_const_b

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.srt.layers.moe.topk.fused_topk`
- `sglang.srt.layers.moe.topk.select_experts`

**3.3% of total serving GPU time** on `stepfun-ai/Step-3.5-Flash` (cookbook-aligned
profile, peak `sharegpt_high`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `void_moe_top_k_256_float_const_b`, category `moe`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `stepfun-ai/Step-3.5-Flash` TP=4 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
