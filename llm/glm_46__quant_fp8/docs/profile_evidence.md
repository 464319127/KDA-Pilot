# Profile evidence - glm_46__quant_fp8

**Standalone kernel target: 6.4% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-4.6-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `zai-org/GLM-4.6-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-4.6-FP8` (slug `glm_46`, tp=8)
- Python interface(s): `sglang.srt.layers.quantization.fp8_kernel.scaled_fp8_quant`, `sglang.srt.layers.quantization.fp8_kernel.sgl_per_token_quant_fp8`
- Kernel family: `quant_fp8`  .  Category: `quant_gemm`
- GPU kernel(s): `void per_token_quant_fp8_kernel<__nv_bfloat16, __nv_fp8_e4m3, 8, 16, false>(__nv_bfloat16 `, `void per_token_quant_fp8_small_batch_kernel<__nv_bfloat16, __nv_fp8_e4m3, 16>(__nv_bfloat1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 6.42% |
| random | conc 100 | 2.58% |
| sharegpt | conc 1 | 6.15% |
| sharegpt | conc 32 | 2.22% |
| sharegpt | conc 100 | 3.10% |

**Peak: 6.4% in `random_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 21
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real zai-org/GLM-4.6-FP8 SGLang TP=8 server in temporary container sglang-glm46. Used a model-local HF cache in offline mode after snapshot download, trust_remote_code, reasoning_parser=glm45, tool_call_parser=glm45 (server_args normalized it to glm; chat template auto-detected glm45), attention_backend=fa4, moe_runner_backend=triton for the Triton MoE tasks, CUDA graph prefill/decode disabled, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `sglang.srt.layers.quantization.fp8_kernel.scaled_fp8_quant`
- `sglang.srt.layers.quantization.fp8_kernel.sgl_per_token_quant_fp8`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
