# KDA Prompt: glm_46__fused_moe_triton

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `<confirm via capture; profiler family=fused_moe_triton>`

**34.8% of total serving GPU time** on `zai-org/GLM-4.6-FP8` (cookbook-aligned profile, peak
`random_high`) — a serving-profile headroom signal used to select this standalone kernel task. Family
`fused_moe_triton`, category `moe`. Triton MoE expert-GEMM (sglang's own fused_experts/fused_moe kernel) — single-GPU optimizable; NOT the comm-fused trtllm MoE path (excluded).

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
