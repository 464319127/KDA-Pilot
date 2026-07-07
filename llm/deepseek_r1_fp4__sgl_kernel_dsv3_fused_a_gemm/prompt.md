# KDA Prompt: deepseek_r1_fp4__sgl_kernel_dsv3_fused_a_gemm

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `sgl_kernel.dsv3_fused_a_gemm`

**42.4% of total serving GPU time** on `nvidia/DeepSeek-R1-0528-FP4-v2` (cookbook-aligned profile, peak
`random_high`) — a serving-profile headroom signal used to select this standalone kernel task. Family
`linear_gemm`, category `quant_gemm`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
