# Profile evidence - mistral_small4__quant_fp8

**Standalone kernel target: 7.1% of total serving GPU time** (max across scenarios) on
`mistralai/Mistral-Small-4-119B-2603`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `mistralai/Mistral-Small-4-119B-2603` production-path capture and replace the old noisy profiler shape strings.

- Model: `mistralai/Mistral-Small-4-119B-2603` (slug `mistral_small4`, tp=1)
- Python interface(s): `sglang.srt.layers.quantization.fp8_kernel.scaled_fp8_quant`, `sglang.srt.layers.quantization.fp8_kernel.static_quant_fp8`
- Kernel family: `quant_fp8`  .  Category: `quant_gemm`
- GPU kernel(s): `_static_quant_fp8`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 6.85% |
| random | conc 100 | 2.14% |
| sharegpt | conc 1 | 7.07% |
| sharegpt | conc 32 | 2.83% |
| sharegpt | conc 100 | 2.35% |

**Peak: 7.1% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 139
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real mistralai/Mistral-Small-4-119B-2603 TP=1 SGLang production-path execution in temporary container sglang-mistral-small4 with TRTLLM/MLA attention and FP8 GEMM; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; request windows include long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `sglang.srt.layers.quantization.fp8_kernel.scaled_fp8_quant`
- `sglang.srt.layers.quantization.fp8_kernel.static_quant_fp8`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
