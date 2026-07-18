# Profile evidence - nemotron3_nano__fused_add_rmsnorm

**Standalone kernel target: 4.4% of total serving GPU time** (max across scenarios) on
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (slug `nemotron3_nano`, tp=1)
- Python interface(s): `sglang.srt.layers.layernorm.fused_add_rmsnorm`
- Kernel family: `None`  .  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 4.41% |
| sharegpt | conc 1 | 4.44% |

**Peak: 4.4% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 4
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 TP=1 SGLang production-path execution in the prepared sglang_bbuf_b300 container; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; capture used temporary import/decomposition compatibility shims for the local Torch/Transformers environment, with request windows covering long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `sglang.srt.layers.layernorm.fused_add_rmsnorm`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
