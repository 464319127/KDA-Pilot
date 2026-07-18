# KDA Prompt: lfm25__sglang_run_activation_inplace

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.jit_kernel.activation._run_activation_inplace`
- `sglang.jit_kernel.activation.run_activation`
- `sglang.jit_kernel.activation.silu_and_mul`

**4.3% of total serving GPU time** on `LiquidAI/LFM2.5-8B-A1B` (cookbook-aligned
profile, peak `random_high`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `None`, category `other`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `LiquidAI/LFM2.5-8B-A1B` TP=1 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
