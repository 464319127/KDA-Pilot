# KDA Prompt: qwen36__fused_add_rmsnorm

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `<confirm via capture; profiler family=fused_add_rmsnorm>`

**3.5% of total serving GPU time** on `Qwen/Qwen3.6-35B-A3B-FP8` (cookbook-aligned profile, peak
`sharegpt_low`) — a serving-profile headroom signal used to select this standalone kernel task. Family
`fused_add_rmsnorm`, category `gemm`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
