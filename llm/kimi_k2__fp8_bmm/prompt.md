# KDA Prompt: kimi_k2__fp8_bmm

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `torch.bmm`

**38.5% of total serving GPU time** on `moonshotai/Kimi-K2-Instruct` (cookbook-aligned
profile, peak `sharegpt_high`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `fp8_bmm`, category `quant_gemm`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `moonshotai/Kimi-K2-Instruct` TP=8 SGLang capture. Normal
RLCR kernel work must not depend on starting SGLang serve, `run_capture`, 8-GPU availability,
or a multi-GPU e2e A/B; optimize and validate via the task-local standalone benchmark on
one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
