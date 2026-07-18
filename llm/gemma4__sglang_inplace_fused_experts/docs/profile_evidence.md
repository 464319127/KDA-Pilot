# Profile evidence - gemma4__sglang_inplace_fused_experts

**Standalone kernel target: 50.8% of total serving GPU time** (max across scenarios) on
`google/gemma-4-26B-A4B-it`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `google/gemma-4-26B-A4B-it` production-path capture and replace the old noisy profiler shape strings.

- Model: `google/gemma-4-26B-A4B-it` (slug `gemma4`, tp=1)
- Python interface(s): `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._fused_moe_kernel_sequence`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts_impl`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels.invoke_fused_moe_kernel`
- Kernel family: `fused_moe_triton`  .  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 22.87% |
| random | conc 32 | 45.23% |
| random | conc 100 | 50.81% |
| sharegpt | conc 1 | 21.68% |
| sharegpt | conc 32 | 43.37% |
| sharegpt | conc 100 | 50.02% |

**Peak: 50.8% in `random_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 18
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real google/gemma-4-26B-A4B-it TP=1 SGLang production-path execution in temporary container sglang-gemma4 on one B300 GPU. Used a model-local HF cache after snapshot download, cookbook-aligned reasoning_parser=gemma4 and tool_call_parser=gemma4, disabled CUDA graph prefill/decode, delayed torch-op wrapping until after imports, preserved server startup/warmup records, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._fused_moe_kernel_sequence`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts_impl`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels.invoke_fused_moe_kernel`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
