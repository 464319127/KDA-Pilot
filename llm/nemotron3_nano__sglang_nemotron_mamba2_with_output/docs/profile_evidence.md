# Profile evidence - nemotron3_nano__sglang_nemotron_mamba2_with_output

**Standalone kernel target: 55.8% of total serving GPU time** (max across scenarios) on
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (slug `nemotron3_nano`, tp=1)
- Python interface(s): `sglang.srt.layers.attention.mamba.causal_conv1d_triton.causal_conv1d_fn`, `sglang.srt.layers.attention.mamba.causal_conv1d_triton.causal_conv1d_update`, `sglang.srt.layers.attention.mamba.ops.ssd_chunk_scan._chunk_scan_fwd`, `sglang.srt.layers.attention.mamba.ops.ssd_chunk_state._chunk_state_fwd`, `sglang.srt.layers.attention.mamba.ops.ssd_chunk_state.chunk_state_varlen`, `sglang.srt.layers.attention.mamba.ops.ssd_combined._mamba_chunk_scan_combined_fwd`, `sglang.srt.layers.attention.mamba.ops.ssd_combined.mamba_chunk_scan_combined`, `sglang.srt.layers.attention.mamba.ops.ssd_state_passing._state_passing_fwd`, `sglang.srt.models.nemotron_h.NemotronHMambaDecoderLayer._forward_mamba`
- Kernel family: `None`  .  Category: `other`
- GPU kernel(s): `_chunk_scan_fwd_kernel`, `_chunk_state_fwd_kernel`, `_chunk_state_varlen_kernel`, `_state_passing_fwd_kernel`, `_static_quant_fp8`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x32x256u2_s5_et128x32_m128x32x32_c1x1x1_rM_TN_transOut_schP`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s8_et128x64_m256x64x32_c2x1x1_rM_TN_transOut_schPd2`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schPd2x`, `bmm_E4m3_E4m3E4m3_Fp32_BtokBfloat16_t128x32x256_s5_et128x32_m128x32x32_c1x1x1_rM_TN_transO`, `bmm_E4m3_E4m3E4m3_Fp32_BtokBfloat16_t128x64x256_s5_et128x64_m256x64x32_c2x1x1_rM_TN_transO`, `bmm_E4m3_E4m3E4m3_Fp32_BtokBfloat16_t128x8x256_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 22.73% |
| random | conc 32 | 47.30% |
| random | conc 100 | 30.55% |
| sharegpt | conc 1 | 17.64% |
| sharegpt | conc 32 | 55.76% |
| sharegpt | conc 100 | 47.84% |

**Peak: 55.8% in `sharegpt_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 341
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 TP=1 SGLang production-path execution in the prepared sglang_bbuf_b300 container; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; capture used temporary import/decomposition compatibility shims for the local Torch/Transformers environment, with request windows covering long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `sglang.srt.layers.attention.mamba.causal_conv1d_triton.causal_conv1d_fn`
- `sglang.srt.layers.attention.mamba.causal_conv1d_triton.causal_conv1d_update`
- `sglang.srt.layers.attention.mamba.ops.ssd_chunk_scan._chunk_scan_fwd`
- `sglang.srt.layers.attention.mamba.ops.ssd_chunk_state._chunk_state_fwd`
- `sglang.srt.layers.attention.mamba.ops.ssd_chunk_state.chunk_state_varlen`
- `sglang.srt.layers.attention.mamba.ops.ssd_combined._mamba_chunk_scan_combined_fwd`
- `sglang.srt.layers.attention.mamba.ops.ssd_combined.mamba_chunk_scan_combined`
- `sglang.srt.layers.attention.mamba.ops.ssd_state_passing._state_passing_fwd`
- `sglang.srt.models.nemotron_h.NemotronHMambaDecoderLayer._forward_mamba`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
