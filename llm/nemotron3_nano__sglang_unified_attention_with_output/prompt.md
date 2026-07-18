# KDA Prompt: nemotron3_nano__sglang_unified_attention_with_output

Target GPU: NVIDIA B200. Optimize the SGLang kernel path behind:

- `sglang.srt.models.nemotron_h.NemotronHAttention.forward`

**6.1% of total serving GPU time** on `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (cookbook-aligned
profile, peak `sharegpt_mid`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `None`, category `quant_gemm`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` TP=1 production-path capture. Normal
RLCR kernel work is a standalone single-GPU task: optimize and validate via the
task-local benchmark on one idle target GPU, without adding external
runtime-readiness or fleet-level A/B gates. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
