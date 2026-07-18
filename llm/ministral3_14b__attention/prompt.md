# KDA Prompt: ministral3_14b__attention

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `flashinfer.decode.trtllm_batch_decode_with_kv_cache`
- `flashinfer.prefill.trtllm_batch_context_with_kv_cache`
- `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_decode`
- `sglang.srt.layers.attention.trtllm_mha_backend.TRTLLMHAAttnBackend.forward_extend`

**18.5% of total serving GPU time** on `mistralai/Ministral-3-14B-Instruct-2512` (cookbook-aligned
profile, peak `sharegpt_high`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `attention`, category `attention`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `mistralai/Ministral-3-14B-Instruct-2512` TP=1 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
