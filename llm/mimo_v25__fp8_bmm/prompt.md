# KDA Prompt: mimo_v25__fp8_bmm

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.forward_decode`
- `sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.forward_extend`
- `sglang.srt.layers.attention.flashattention_backend.flash_attn_with_kvcache`

**27.9% of total serving GPU time** on `XiaomiMiMo/MiMo-V2.5` (cookbook-aligned
profile, peak `sharegpt_mid`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `fp8_bmm`, category `quant_gemm`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `XiaomiMiMo/MiMo-V2.5` TP=4 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
