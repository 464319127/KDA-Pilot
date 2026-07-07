# Profile evidence — deepseek_v3__fp8_bmm

**Standalone kernel target: 7.4% of total serving GPU time** (max across scenarios) on
`deepseek-ai/DeepSeek-V3`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `deepseek-ai/DeepSeek-V3` (slug `deepseek_v3`, tp=8)
- Python interface: `<confirm via capture; profiler family=fp8_bmm>`
- Kernel family: `fp8_bmm`  ·  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_E4m3E4m3_Fp32_t128x16x128_s6_et64x16_m64x16x32_c1x1x1_rM_TN_transOut_noShfl_d`, `bmm_E4m3_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_TN_transOut_noShfl_dsF`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 7.36% |
| sharegpt | conc 32 | 4.03% |

**Peak: 7.4% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[128], [128], []]`
- `[[1]]`
- `[[32], []]`
- `[[32]]`
- `[[6489], [], [], []]`
- `[[962], [], [], [], []]`
- `[[962], [], [], []]`
- `[[[128], [18]], []]`
- `[[[448]], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path deepseek-ai/DeepSeek-V3 --tp 8 --speculative-algorithm EAGLE
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
