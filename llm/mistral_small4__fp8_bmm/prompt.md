# KDA Prompt: mistral_small4__fp8_bmm

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sgl_kernel.gemm.bmm_fp8`
- `sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla._bmm_fp8_op`
- `sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla.bmm_fp8`
- `torch.ops.sgl_kernel.bmm_fp8.default`

**42.3% of total serving GPU time** on `mistralai/Mistral-Small-4-119B-2603` (cookbook-aligned
profile, peak `random_high`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `fp8_bmm`, category `other`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `mistralai/Mistral-Small-4-119B-2603` TP=1 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
