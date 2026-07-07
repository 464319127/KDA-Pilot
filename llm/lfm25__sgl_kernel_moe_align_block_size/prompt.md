# KDA Prompt: lfm25__sgl_kernel_moe_align_block_size

Target GPU: NVIDIA B200.

Optimize the SGLang kernel behind the Python interface:

- `sgl_kernel.moe_align_block_size`

**This kernel is 5.1% of total serving GPU time** on `LiquidAI/LFM2.5-8B-A1B` (cookbook-aligned
profile, peak in `random_low`) — a serving-profile headroom signal used to select this standalone kernel task. Category: `moe`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU breakdown, the GPU
kernel(s) involved, captured input shapes, and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow `llm/docs/llm_kernel_optimization_rules.md`
(CUDA, no DSL) and `llm/docs/llm_correctness_contract.md`.
