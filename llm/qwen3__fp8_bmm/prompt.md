# KDA Prompt: qwen3__fp8_bmm

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `flashinfer.decode.trtllm_batch_decode_with_kv_cache`
- `flashinfer.prefill.trtllm_batch_context_with_kv_cache`

**32.0% of total serving GPU time** on `Qwen/Qwen3-235B-A22B-Instruct-2507` (cookbook-aligned
profile, peak `random_mid`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `fp8_bmm`, category `other`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `Qwen/Qwen3-235B-A22B-Instruct-2507` TP=8 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
