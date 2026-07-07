# KDA Prompt: nemotron3_nano__fused_add_rmsnorm

Target GPU: NVIDIA B200.

Optimize the SGLang kernel behind the Python interface:

- `<confirm via capture; profiler role=fused_add_rmsnorm>`

**This kernel is 4.4% of total serving GPU time** on `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (cookbook-aligned
profile, peak in `sharegpt_low`) — a serving-profile headroom signal used to select this standalone kernel task. Category: `gemm`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU breakdown, the GPU
kernel(s) involved, captured input shapes, and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow `llm/docs/llm_kernel_optimization_rules.md`
(CUDA, no DSL) and `llm/docs/llm_correctness_contract.md`.
