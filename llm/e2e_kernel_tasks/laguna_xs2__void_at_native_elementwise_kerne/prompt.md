# KDA Prompt: laguna_xs2__void_at_native_elementwise_kerne

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `<confirm via capture; profiler family=void_at_native_elementwise_kerne>`

**4.6% of total GPU time** on `poolside/Laguna-XS.2-NVFP4` (cookbook-aligned profile, peak
`random_mid`) — a genuine end-to-end target selected by profiler e2e share. Family
`void_at_native_elementwise_kerne`, category `memory_bound`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
the cookbook deployment, and the scenario to re-run for the e2e A/B. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
