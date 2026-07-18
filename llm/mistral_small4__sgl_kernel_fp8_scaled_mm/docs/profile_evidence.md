# Profile evidence - mistral_small4__sgl_kernel_fp8_scaled_mm

**Standalone kernel target: 28.4% of total serving GPU time** (max across scenarios) on
`mistralai/Mistral-Small-4-119B-2603`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `mistralai/Mistral-Small-4-119B-2603` production-path capture and replace the old noisy profiler shape strings.

- Model: `mistralai/Mistral-Small-4-119B-2603` (slug `mistral_small4`, tp=1)
- Python interface(s): `sglang.srt.layers.quantization.fp8.Fp8LinearMethod.apply`, `sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear`, `torch.ops.sgl_kernel.fp8_scaled_mm.default`
- Kernel family: `linear_gemm`  .  Category: `gemm`
- GPU kernel(s): `_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10colle`, `nvjet_sm100_tst_128x8_64x12_2x1_v_bz_TNT`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 26.95% |
| random | conc 32 | 15.26% |
| random | conc 100 | 10.18% |
| sharegpt | conc 1 | 28.39% |
| sharegpt | conc 32 | 12.38% |
| sharegpt | conc 100 | 9.40% |

**Peak: 28.4% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 42
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real mistralai/Mistral-Small-4-119B-2603 TP=1 SGLang production-path execution in temporary container sglang-mistral-small4 with TRTLLM/MLA attention and FP8 GEMM; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; request windows include long/short prompt and mid/high concurrency serving paths.

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
