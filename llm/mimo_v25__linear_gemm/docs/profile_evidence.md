# Profile evidence - mimo_v25__linear_gemm

**Standalone kernel target: 9.2% of total serving GPU time** (max across scenarios) on
`XiaomiMiMo/MiMo-V2.5`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `XiaomiMiMo/MiMo-V2.5` production-path capture and replace the old noisy profiler shape strings.

- Model: `XiaomiMiMo/MiMo-V2.5` (slug `mimo_v25`, tp=4)
- Python interface(s): `sglang.srt.layers.quantization.fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback`, `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `gemm`
- GPU kernel(s): `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_tnn_align1_bias_f32_relu`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu`, `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x`, `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 8.79% |
| random | conc 100 | 5.91% |
| sharegpt | conc 1 | 2.07% |
| sharegpt | conc 32 | 9.17% |
| sharegpt | conc 100 | 2.61% |

**Peak: 9.2% in `sharegpt_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 32
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real XiaomiMiMo/MiMo-V2.5 TP=4 SGLang production-path execution in temporary container sglang-mimo-v25 with FA4 attention and flashinfer_trtllm MoE; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; request windows include long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `sglang.srt.layers.quantization.fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback`
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
