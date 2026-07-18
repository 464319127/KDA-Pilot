# Profile evidence - lfm25__sglang_run_activation_inplace

**Standalone kernel target: 4.3% of total serving GPU time** (max across scenarios) on
`LiquidAI/LFM2.5-8B-A1B`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `LiquidAI/LFM2.5-8B-A1B` production-path capture and replace the old noisy profiler shape strings.

- Model: `LiquidAI/LFM2.5-8B-A1B` (slug `lfm25`, tp=1)
- Python interface(s): `sglang.jit_kernel.activation._run_activation_inplace`, `sglang.jit_kernel.activation.run_activation`, `sglang.jit_kernel.activation.silu_and_mul`
- Kernel family: `None`  .  Category: `other`
- GPU kernel(s): `void (anonymous namespace)::act_and_mul_kernel<__nv_bfloat16, ((anonymous namespace)::Acti`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.94% |
| random | conc 32 | 3.54% |
| random | conc 100 | 4.35% |
| sharegpt | conc 1 | 3.16% |
| sharegpt | conc 32 | 3.51% |
| sharegpt | conc 100 | 3.73% |

**Peak: 4.3% in `random_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 60
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real LiquidAI/LFM2.5-8B-A1B TP=1 SGLang production-path execution in temporary container sglang-lfm25 on a single B300 GPU. Used a model-local HF cache after snapshot download, attention_backend=flashinfer, reasoning_parser=qwen3, tool_call_parser=lfm2, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

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
