# Profile evidence - glm_46__void_at_native_sbtopk_gather_top

**Standalone kernel target: 6.1% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-4.6-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below were recaptured from a real `zai-org/GLM-4.6-FP8` server run and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-4.6-FP8` (slug `glm_46`, tp=8)
- Python interface(s): `sglang.srt.layers.moe.topk.biased_grouped_topk_impl`, `sglang.srt.layers.moe.topk.select_experts`
- Kernel family: `void_at_native_sbtopk_gather_top`  .  Category: `moe`
- GPU kernel(s): `void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::Tenso`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 6.07% |
| sharegpt | conc 100 | 5.32% |

**Peak: 6.1% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 810
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real zai-org/GLM-4.6-FP8 SGLang TP=8 server in temporary container sglang-glm46. Used a model-local HF cache in offline mode after snapshot download, trust_remote_code, reasoning_parser=glm45, tool_call_parser=glm45 (server_args normalized it to glm; chat template auto-detected glm45), attention_backend=fa4, moe_runner_backend=triton for the Triton MoE tasks, CUDA graph prefill/decode disabled, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `sglang.srt.layers.moe.topk.biased_grouped_topk_impl`
- `sglang.srt.layers.moe.topk.select_experts`

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
