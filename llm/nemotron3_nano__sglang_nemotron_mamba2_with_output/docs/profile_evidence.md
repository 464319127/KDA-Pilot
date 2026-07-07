# Profile evidence — nemotron3_nano__sglang_nemotron_mamba2_with_output

**Why this is a standalone kernel target:** it is **55.8% of total serving GPU time** (max across scenarios) on `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`, measured by profiling the exact
cookbook deployment. This is target-selection provenance and headroom context, not the validation path. Clean Python interface from profiler provenance.

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (slug `nemotron3_nano`, tp=1)
- Python interface: `sglang.nemotron_mamba2_with_output`
- Category: `other`
- GPU kernel(s): `_chunk_scan_fwd_kernel`, `_chunk_state_fwd_kernel`, `_chunk_state_varlen_kernel`, `_state_passing_fwd_kernel`, `_static_quant_fp8`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x32x256u2_s5_et128x32_m128x32x32_c1x1x1_rM_TN_transOut_schP`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s8_et128x64_m256x64x32_c2x1x1_rM_TN_transOut_schPd2`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x256u2_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schPd2x`, `bmm_E4m3_E4m3E4m3_Fp32_BtokBfloat16_t128x32x256_s5_et128x32_m128x32x32_c1x1x1_rM_TN_transO`, `bmm_E4m3_E4m3E4m3_Fp32_BtokBfloat16_t128x64x256_s5_et128x64_m256x64x32_c2x1x1_rM_TN_transO`, `bmm_E4m3_E4m3E4m3_Fp32_BtokBfloat16_t128x8x256_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_`
- Profiler op provenance: `aten::_local_scalar_dense`, `aten::any`, `aten::as_strided`, `aten::copy_`, `aten::empty`, `aten::empty_like`

## % of GPU time by scenario

| dataset | scenario (concurrency) | % of GPU time |
|---|---|---|
| random_low | random (conc 1) | 22.73% |
| random_mid | random (conc 32) | 47.30% |
| random_high | random (conc 100) | 30.55% |
| sharegpt_low | sharegpt (conc 1) | 17.64% |
| sharegpt_mid | sharegpt (conc 32) | 55.76% |
| sharegpt_high | sharegpt (conc 100) | 47.84% |

**Peak: 55.8% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[1024, 2688], [1024, 2688], []]`
- `[[103, 2688], [2688, 10304]]`
- `[[10304, 2688]]`
- `[[112, 2688], [112, 2688], []]`
- `[[13312, 2688], [13312, 2688], []]`
- `[[1553, 2688], [2688, 10304]]`
- `[[1553, 2688], [], [], [], [], []]`
- `[[1792, 2688], [1792, 2688], []]`
- `[[1], [1], []]`
- `[[1]]`
- `[[23, 1246, 6144, 3], [], []]`
- `[[23, 1246, 64, 64, 128], [], [], []]`

## Original serving capture command (provenance only)
```bash
python3 -m sglang.launch_server --model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --trust-remote-code --max-running-requests 1024
```

Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
