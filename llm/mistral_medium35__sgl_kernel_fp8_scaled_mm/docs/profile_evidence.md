# Profile evidence - mistral_medium35__sgl_kernel_fp8_scaled_mm

**Standalone kernel target: 63.5% of total serving GPU time** (max across scenarios) on
`mistralai/Mistral-Medium-3.5-128B`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `mistralai/Mistral-Medium-3.5-128B` production-path capture and replace the old noisy profiler shape strings.

- Model: `mistralai/Mistral-Medium-3.5-128B` (slug `mistral_medium35`, tp=2)
- Python interface(s): `sglang.srt.layers.quantization.fp8.Fp8LinearMethod.apply`, `sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear`, `torch.ops.sgl_kernel.fp8_scaled_mm.default`
- Kernel family: `linear_gemm`  .  Category: `gemm`
- GPU kernel(s): `_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10colle`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 50.27% |
| random | conc 32 | 9.61% |
| random | conc 100 | 56.06% |
| sharegpt | conc 1 | 49.77% |
| sharegpt | conc 32 | 63.53% |
| sharegpt | conc 100 | 40.95% |

**Peak: 63.5% in `sharegpt_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 24
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real mistralai/Mistral-Medium-3.5-128B TP=2 SGLang production-path execution in temporary container sglang-mistral-medium35 with TRTLLM MHA attention and FP8 GEMM; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; request windows include long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `sglang.srt.layers.quantization.fp8.Fp8LinearMethod.apply`
- `sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear`
- `torch.ops.sgl_kernel.fp8_scaled_mm.default`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
