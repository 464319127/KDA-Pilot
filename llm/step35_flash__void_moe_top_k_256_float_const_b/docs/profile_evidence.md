# Profile evidence - step35_flash__void_moe_top_k_256_float_const_b

**Standalone kernel target: 3.3% of total serving GPU time** (max across scenarios) on
`stepfun-ai/Step-3.5-Flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `stepfun-ai/Step-3.5-Flash` production-path capture and replace the old noisy profiler shape strings.

- Model: `stepfun-ai/Step-3.5-Flash` (slug `step35_flash`, tp=4)
- Python interface(s): `sglang.srt.layers.moe.topk.fused_topk`, `sglang.srt.layers.moe.topk.select_experts`
- Kernel family: `void_moe_top_k_256_float_const_b`  .  Category: `moe`
- GPU kernel(s): `void moeTopK<256>(float const*, bool const*, float*, int*, int, int, int, int, bool, float`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.79% |
| random | conc 32 | 3.09% |
| random | conc 100 | 2.85% |
| sharegpt | conc 1 | 3.00% |
| sharegpt | conc 32 | 2.88% |
| sharegpt | conc 100 | 3.28% |

**Peak: 3.3% in `sharegpt_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 129
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real stepfun-ai/Step-3.5-Flash TP=4 SGLang production-path execution in temporary container sglang-step35-flash with FA4 attention; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; request windows include long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `sglang.srt.layers.moe.topk.fused_topk`
- `sglang.srt.layers.moe.topk.select_experts`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
