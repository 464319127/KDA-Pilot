# Profile evidence - glm_46__attention

**Standalone kernel target: 9.3% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-4.6-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below were recaptured from a real `zai-org/GLM-4.6-FP8` server run and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-4.6-FP8` (slug `glm_46`, tp=8)
- Python interface(s): `sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.forward_decode`, `sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.forward_extend`, `sglang.srt.layers.attention.flashattention_backend.flash_attn_with_kvcache`
- Kernel family: `attention`  .  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__t`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.31% |
| random | conc 32 | 3.66% |
| random | conc 100 | 4.22% |
| sharegpt | conc 1 | 9.32% |
| sharegpt | conc 32 | 6.87% |
| sharegpt | conc 100 | 7.67% |

**Peak: 9.3% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 19
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real zai-org/GLM-4.6-FP8 SGLang TP=8 server in temporary container sglang-glm46. Used a model-local HF cache in offline mode after snapshot download, trust_remote_code, reasoning_parser=glm45, tool_call_parser=glm45 (server_args normalized it to glm; chat template auto-detected glm45), attention_backend=fa4, moe_runner_backend=triton for the Triton MoE tasks, CUDA graph prefill/decode disabled, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.forward_decode`
- `sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.forward_extend`
- `sglang.srt.layers.attention.flashattention_backend.flash_attn_with_kvcache`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Original serving profile command (provenance only)
```bash
sglang serve --model-path zai-org/GLM-4.6-FP8 --tp 8 --reasoning-parser glm45 --tool-call-parser glm45 --attention-backend fa4
```
This command is retained only to explain target selection. Normal RLCR kernel
work must not depend on a live SGLang server, `run_capture`, 8-GPU availability,
or a multi-GPU e2e gate. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set. Re-run serving capture only
when intentionally refreshing these evidence files.
