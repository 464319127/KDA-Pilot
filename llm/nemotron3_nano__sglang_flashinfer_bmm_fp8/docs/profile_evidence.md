# Profile evidence - nemotron3_nano__sglang_flashinfer_bmm_fp8

**Standalone kernel target: 20.7% of total serving GPU time** (max across scenarios) on
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (slug `nemotron3_nano`, tp=1)
- Python interface(s): `flashinfer.gemm.bmm_fp8`, `sglang.srt.layers.quantization.fp8_utils.flashinfer_bmm_fp8`
- Kernel family: `None`  .  Category: `gemm`
- GPU kernel(s): `_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10colle`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s8_et128x64_m256x64x32_c2x1x1_rM_TN_transOut_schPd2`, `bmm_E4m3_E4m3E4m3_Fp32_BtokBfloat16_t128x64x256_s5_et128x64_m256x64x32_c2x1x1_rM_TN_transO`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 20.68% |
| random | conc 100 | 10.70% |
| sharegpt | conc 32 | 8.33% |
| sharegpt | conc 100 | 3.78% |

**Peak: 20.7% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 24
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 TP=1 SGLang production-path execution in the prepared sglang_bbuf_b300 container; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; capture used temporary import/decomposition compatibility shims for the local Torch/Transformers environment, with request windows covering long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `flashinfer.gemm.bmm_fp8`
- `sglang.srt.layers.quantization.fp8_utils.flashinfer_bmm_fp8`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
