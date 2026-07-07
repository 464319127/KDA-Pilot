# KDA Prompt: ministral3_14b__sglang_run_activation_inplace

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `sglang._run_activation_inplace`

**5.3% of total serving GPU time** on `mistralai/Ministral-3-14B-Instruct-2512` (cookbook-aligned profile, peak
`random_mid`) — a serving-profile headroom signal used to select this standalone kernel task. Family
`activation`, category `other`. Activation (SiLU/GELU+mul). Prior guidance: limited headroom — deprioritize.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
