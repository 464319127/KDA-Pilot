# KDA Prompt: ministral3_14b__fused_add_rmsnorm

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.srt.layers.layernorm.fused_add_rmsnorm`

**5.8% of total serving GPU time** on `mistralai/Ministral-3-14B-Instruct-2512` (cookbook-aligned
profile, peak `random_low`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `fused_add_rmsnorm`, category `gemm`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `mistralai/Ministral-3-14B-Instruct-2512` TP=1 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
