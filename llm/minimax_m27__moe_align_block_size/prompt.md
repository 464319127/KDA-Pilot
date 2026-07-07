# KDA Prompt: minimax_m27__moe_align_block_size

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `<confirm via capture; profiler family=moe_align_block_size>`

**3.6% of total serving GPU time** on `MiniMaxAI/MiniMax-M2.7` (cookbook-aligned profile, peak
`sharegpt_low`) — a serving-profile headroom signal used to select this standalone kernel task. Family
`moe_align_block_size`, category `moe`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
