# Profile evidence - step35_flash__linear_gemm

**Standalone kernel target: 6.9% of total serving GPU time** (max across scenarios) on
`stepfun-ai/Step-3.5-Flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `stepfun-ai/Step-3.5-Flash` production-path capture and replace the old noisy profiler shape strings.

- Model: `stepfun-ai/Step-3.5-Flash` (slug `step35_flash`, tp=4)
- Python interface(s): `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `quant_gemm`
- GPU kernel(s): `nvjet_tst_32x64_64x16_4x1_v_bz_TNN`, `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.26% |
| random | conc 32 | 5.82% |
| random | conc 100 | 5.37% |
| sharegpt | conc 1 | 6.29% |
| sharegpt | conc 32 | 6.04% |
| sharegpt | conc 100 | 6.87% |

**Peak: 6.9% in `sharegpt_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 20
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real stepfun-ai/Step-3.5-Flash TP=4 SGLang production-path execution in temporary container sglang-step35-flash with FA4 attention; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; request windows include long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
