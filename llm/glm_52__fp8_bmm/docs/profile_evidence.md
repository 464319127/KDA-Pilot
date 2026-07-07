# Profile evidence — glm_52__fp8_bmm

**Standalone kernel target: 25.0% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-5.2-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below were recaptured from a real GLM-5.2-FP8 TP=8 server run and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-5.2-FP8` (slug `glm_52`, tp=8)
- Python interface(s): `torch.bmm`
- Kernel family: `fp8_bmm`  ·  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_TN_transOut_noShfl`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x32x128_s6_et64x32_m64x32x32_c1x1x1_rM_TN_transOut_noShfl_d`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s6_et64x64_m64x64x32_c1x1x1_rM_TN_transOut_noShfl_d`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128u2_s6_et64x64_m64x64x32_c1x1x1_rM_TN_transOut_noShfl`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_TN_transOut_noShfl_ds`, `bmm_E4m3_E4m3E4m3_Fp32_t128x128x128u2_s5_et64x128_m64x128x32_c1x1x1_rM_TN_transOut_noShfl_`, `bmm_E4m3_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_TN_transOut_noShfl_dsF`, `bmm_E4m3_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_c1x1x1_rM_TN_transOut_noShfl_dsF`, `bmm_E4m3_E4m3E4m3_Fp32_t128x64x128u2_s6_et64x64_m64x64x32_c1x1x1_rM_TN_transOut_noShfl_dsF`, `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_TN_transOut_noShfl_dsFp8_`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 10.99% |
| random | conc 100 | 21.77% |
| sharegpt | conc 1 | 9.37% |
| sharegpt | conc 32 | 25.02% |
| sharegpt | conc 100 | 16.33% |

**Peak: 25.0% in `sharegpt_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 10
- Capture note: Captured on verda-b300-fin-03-1/light-face-hides-fin-03-1, sglang-glm52 container, zai-org/GLM-5.2-FP8 tp=8, CUDA graphs disabled for Python API capture, 2026-07-07. Final records include DSA/TRTLLM attention, torch.bmm, DeepGEMM, and per-token quant call contracts from real server requests.

Functions covered:
- `torch.bmm`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path zai-org/GLM-5.2-FP8 --tp 8 --trust-remote-code --mem-fraction-static 0.8
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
