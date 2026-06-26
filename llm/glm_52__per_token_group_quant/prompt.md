# KDA Prompt: glm_52__per_token_group_quant

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `<confirm via capture; profiler family=per_token_group_quant>`

**10.9% of total GPU time** on `zai-org/GLM-5.2-FP8` (cookbook-aligned profile, peak
`random_low`) — a genuine end-to-end target selected by profiler e2e share. Family
`per_token_group_quant`, category `quant_gemm`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
the cookbook deployment, and the scenario to re-run for the e2e A/B. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
