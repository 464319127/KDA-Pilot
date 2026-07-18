# Profile evidence - ministral3_14b__sglang_run_activation_inplace

**Standalone kernel target: 5.3% of total serving GPU time** (max across scenarios) on
`mistralai/Ministral-3-14B-Instruct-2512`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `mistralai/Ministral-3-14B-Instruct-2512` production-path capture and replace the old noisy profiler shape strings.

- Model: `mistralai/Ministral-3-14B-Instruct-2512` (slug `ministral3_14b`, tp=1)
- Python interface(s): `sglang.jit_kernel.activation._run_activation_inplace`, `sglang.jit_kernel.activation.run_activation`, `sglang.jit_kernel.activation.silu_and_mul`
- Kernel family: `activation`  .  Category: `other`
- GPU kernel(s): `void (anonymous namespace)::act_and_mul_kernel<__nv_bfloat16, ((anonymous namespace)::Acti`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.84% |
| random | conc 32 | 5.33% |
| random | conc 100 | 4.13% |
| sharegpt | conc 32 | 5.06% |
| sharegpt | conc 100 | 4.13% |

**Peak: 5.3% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 12
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real mistralai/Ministral-3-14B-Instruct-2512 TP=1 SGLang production-path execution in temporary container sglang-ministral3-14b on one B300 GPU. Used a model-local HF cache after snapshot download, cookbook-aligned --trust-remote-code, disabled CUDA graph prefill/decode, enabled SGLANG_DIFFUSION_ENABLE_W8A8_FP8_GEMM=1 to expose the FP8 W8A8 GEMM path selected by the original evidence, delayed torch-op wrapping until after imports, preserved server startup/warmup records for static_quant_fp8 and sgl_kernel fp8_scaled_mm shapes, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `sglang.jit_kernel.activation._run_activation_inplace`
- `sglang.jit_kernel.activation.run_activation`
- `sglang.jit_kernel.activation.silu_and_mul`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
