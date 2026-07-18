# KDA Prompt: ministral3_14b__sgl_kernel_fp8_scaled_mm

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.srt.layers.quantization.fp8.Fp8LinearMethod.apply`
- `sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear`
- `torch.ops.sgl_kernel.fp8_scaled_mm.default`

**69.4% of total serving GPU time** on `mistralai/Ministral-3-14B-Instruct-2512` (cookbook-aligned
profile, peak `random_mid`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `linear_gemm`, category `gemm`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `mistralai/Ministral-3-14B-Instruct-2512` TP=1 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
