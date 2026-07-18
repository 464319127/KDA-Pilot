# Profile evidence - glm_46__void_moe_sum_reduce_warp_per_tok

**Standalone kernel target: 3.4% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-4.6-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `zai-org/GLM-4.6-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-4.6-FP8` (slug `glm_46`, tp=8)
- Python interface(s): `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.moe_sum_reduce`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.moe_sum_reduce_torch_compile`
- Kernel family: `void_moe_sum_reduce_warp_per_tok`  .  Category: `moe`
- GPU kernel(s): `void moe_sum_reduce_warp_per_token_vec_kernel<8>(c10::BFloat16 const*, c10::BFloat16*, lon`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 3.38% |
| random | conc 100 | 2.58% |
| sharegpt | conc 32 | 3.13% |
| sharegpt | conc 100 | 2.13% |

**Peak: 3.4% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 9
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real zai-org/GLM-4.6-FP8 SGLang TP=8 server in temporary container sglang-glm46. Used a model-local HF cache in offline mode after snapshot download, trust_remote_code, reasoning_parser=glm45, tool_call_parser=glm45 (server_args normalized it to glm; chat template auto-detected glm45), attention_backend=fa4, moe_runner_backend=triton for the Triton MoE tasks, CUDA graph prefill/decode disabled, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.moe_sum_reduce`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.moe_sum_reduce_torch_compile`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
