# KDA Prompt: glm_46__fused_moe_triton

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._fused_moe_kernel_sequence`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts_impl`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels.invoke_fused_moe_kernel`

**34.8% of total serving GPU time** on `zai-org/GLM-4.6-FP8` (cookbook-aligned
profile, peak `random_high`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `fused_moe_triton`, category `moe`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `zai-org/GLM-4.6-FP8` TP=8 SGLang capture. Normal
RLCR kernel work must not depend on starting SGLang serve, `run_capture`, 8-GPU availability,
or a multi-GPU e2e A/B; optimize and validate via the task-local standalone benchmark on
one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
