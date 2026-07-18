# Profile evidence - ling_26__sglang_inplace_fused_experts

**Standalone kernel target: 32.3% of total serving GPU time** (max across scenarios) on
`inclusionAI/Ling-2.6-flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `inclusionAI/Ling-2.6-flash` production-path capture and replace the old noisy profiler shape strings.

- Model: `inclusionAI/Ling-2.6-flash` (slug `ling_26`, tp=4)
- Python interface(s): `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._fused_moe_kernel_sequence`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts_impl`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels.invoke_fused_moe_kernel`
- Kernel family: `fused_moe_triton`  .  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 9.39% |
| random | conc 32 | 32.32% |
| random | conc 100 | 18.60% |
| sharegpt | conc 1 | 9.27% |
| sharegpt | conc 32 | 26.15% |
| sharegpt | conc 100 | 18.55% |

**Peak: 32.3% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 18
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real inclusionAI/Ling-2.6-flash TP=4 SGLang production-path execution in temporary container sglang-ling26; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; request windows include long/short prompt and mid/high concurrency serving paths.

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
