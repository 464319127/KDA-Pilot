# KDA Prompt: qwen35__linear_gemm

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `torch.nn.functional.linear`

**40.7% of total serving GPU time** on `nvidia/Qwen3.5-397B-A17B-NVFP4` (cookbook-aligned
profile, peak `sharegpt_low`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `linear_gemm`, category `quant_gemm`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `nvidia/Qwen3.5-397B-A17B-NVFP4` TP=4 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
