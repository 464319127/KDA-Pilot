# KDA Prompt: deepseek_math_v2__void_anonymous_namespace_fast_ha

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `<confirm via capture; profiler family=void_anonymous_namespace_fast_ha>`

**3.8% of total serving GPU time** on `deepseek-ai/DeepSeek-Math-V2` (cookbook-aligned profile, peak
`sharegpt_mid`) — a serving-profile headroom signal used to select this standalone kernel task. Family
`void_anonymous_namespace_fast_ha`, category `other`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
