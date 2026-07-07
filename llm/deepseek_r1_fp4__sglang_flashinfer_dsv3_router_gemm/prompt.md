# KDA Prompt: deepseek_r1_fp4__sglang_flashinfer_dsv3_router_gemm

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `sglang.flashinfer_dsv3_router_gemm`

**19.1% of total serving GPU time** on `nvidia/DeepSeek-R1-0528-FP4-v2` (cookbook-aligned profile, peak
`sharegpt_low`) — a serving-profile headroom signal used to select this standalone kernel task. Family
`attention`, category `gemm`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
