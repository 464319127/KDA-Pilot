# Profile evidence - mimo_v25__fused_add_rmsnorm

**Standalone kernel target: 34.6% of total serving GPU time** (max across scenarios) on
`XiaomiMiMo/MiMo-V2.5`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `XiaomiMiMo/MiMo-V2.5` production-path capture and replace the old noisy profiler shape strings.

- Model: `XiaomiMiMo/MiMo-V2.5` (slug `mimo_v25`, tp=4)
- Python interface(s): `sglang.srt.layers.layernorm.fused_add_rmsnorm`
- Kernel family: `fused_add_rmsnorm`  .  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 31.79% |
| random | conc 32 | 16.86% |
| random | conc 100 | 23.71% |
| sharegpt | conc 1 | 34.62% |
| sharegpt | conc 32 | 14.71% |
| sharegpt | conc 100 | 25.82% |

**Peak: 34.6% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 4
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real XiaomiMiMo/MiMo-V2.5 TP=4 SGLang production-path execution in temporary container sglang-mimo-v25 with FA4 attention and flashinfer_trtllm MoE; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; request windows include long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `sglang.srt.layers.layernorm.fused_add_rmsnorm`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
