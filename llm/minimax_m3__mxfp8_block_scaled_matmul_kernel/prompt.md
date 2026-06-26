# KDA Prompt: minimax_m3__mxfp8_block_scaled_matmul_kernel

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `<confirm via capture; profiler family=mxfp8_block_scaled_matmul_kernel>`

**10.2% of total GPU time** on `MiniMaxAI/MiniMax-M3-MXFP8` (cookbook-aligned profile, peak
`sharegpt_low`) — a genuine end-to-end target selected by profiler e2e share. Family
`mxfp8_block_scaled_matmul_kernel`, category `quant_gemm`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
the cookbook deployment, and the scenario to re-run for the e2e A/B. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
