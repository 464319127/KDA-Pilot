# KDA Prompt: glm_45__linear_gemm

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear`
- `sglang.srt.layers.quantization.fp8_utils.fp8_scaled_mm`
- `torch.nn.functional.linear`

**7.1% of total serving GPU time** on `zai-org/GLM-4.5-FP8` (cookbook-aligned
profile, peak `random_low`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `linear_gemm`, category `gemm`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `zai-org/GLM-4.5-FP8` TP=8 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
