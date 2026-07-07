# KDA Prompt: kimi_k2__sglang_deep_gemm_fp8_fp8_bf16_nt

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.srt.layers.quantization.fp8_kernel.deep_gemm_fp8_fp8_bf16_nt`

**23.1% of total serving GPU time** on `moonshotai/Kimi-K2-Instruct` (cookbook-aligned
profile, peak `sharegpt_low`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `linear_gemm`, category `quant_gemm`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `moonshotai/Kimi-K2-Instruct` TP=8 SGLang capture. Normal
RLCR kernel work must not depend on starting SGLang serve, `run_capture`, 8-GPU availability,
or a multi-GPU e2e A/B; optimize and validate via the task-local standalone benchmark on
one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
