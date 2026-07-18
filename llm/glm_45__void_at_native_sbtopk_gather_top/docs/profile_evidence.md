# Profile evidence - glm_45__void_at_native_sbtopk_gather_top

**Standalone kernel target: 5.9% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-4.5-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `zai-org/GLM-4.5-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-4.5-FP8` (slug `glm_45`, tp=8)
- Python interface(s): `sglang.srt.layers.moe.topk.biased_grouped_topk_impl`, `sglang.srt.layers.moe.topk.select_experts`
- Kernel family: `void_at_native_sbtopk_gather_top`  .  Category: `moe`
- GPU kernel(s): `void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::Tenso`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 5.92% |
| random | conc 100 | 5.84% |
| sharegpt | conc 32 | 5.75% |
| sharegpt | conc 100 | 5.05% |

**Peak: 5.9% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 810
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real zai-org/GLM-4.5-FP8 SGLang TP=8 server in temporary container sglang-glm45. Used local NVMe HF cache in offline mode, trust_remote_code, reasoning_parser=glm45, tool_call_parser=glm45 (runtime normalized to glm), attention_backend=fa4, moe_runner_backend=triton for the Triton MoE tasks, CUDA graph prefill/decode disabled, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `sglang.srt.layers.moe.topk.biased_grouped_topk_impl`
- `sglang.srt.layers.moe.topk.select_experts`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
