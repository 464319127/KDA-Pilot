# KDA Prompt: nemotron3_super__sglang_nemotron_mamba2_with_output

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `sglang.nemotron_mamba2_with_output`

**27.3% of total serving GPU time** on `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (cookbook-aligned profile, peak
`random_high`) — a serving-profile headroom signal used to select this standalone kernel task. Family
`mamba2_ssm`, category `other`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
