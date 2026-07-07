# KDA Prompt: laguna_xs2__void_cvt_fp16_to_fp4_nv_bfloat16

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `<confirm via capture; profiler family=void_cvt_fp16_to_fp4_nv_bfloat16>`

**9.2% of total serving GPU time** on `poolside/Laguna-XS.2-NVFP4` (cookbook-aligned profile, peak
`random_mid`) — a serving-profile headroom signal used to select this standalone kernel task. Family
`void_cvt_fp16_to_fp4_nv_bfloat16`, category `quant_gemm`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
