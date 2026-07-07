# Profile evidence — gpt_oss_120b__linear_gemm

**Standalone kernel target: 29.8% of total serving GPU time** (max across scenarios) on
`openai/gpt-oss-120b`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `openai/gpt-oss-120b` (slug `gpt_oss_120b`, tp=8)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_24x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_bias_TNN`, `nvjet_sm100_tst_32x64_64x16_4x2_2cta_h_bz_splitK_bias_TNN`, `nvjet_sm100_tst_64x16_64x16_2x4_2cta_h_bz_splitK_bias_TNT`, `nvjet_sm100_tst_64x32_64x16_2x4_2cta_h_bz_splitK_bias_TNT`, `nvjet_sm100_tst_64x8_64x16_1x2_h_bz_bias_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_bias_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_splitK_bias_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 27.53% |
| random | conc 32 | 9.67% |
| random | conc 100 | 26.43% |
| sharegpt | conc 1 | 29.79% |
| sharegpt | conc 32 | 17.52% |
| sharegpt | conc 100 | 20.69% |

**Peak: 29.8% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[0], [], [], [], [], [], []]`
- `[[1, 201088], [], [], []]`
- `[[130], [], [], []]`
- `[[192, 1, 64], [], [], []]`
- `[[192], [], [], [], [], [], []]`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[262, 512], []]`
- `[[320], [320], []]`
- `[[320], [], [], []]`
- `[[38], []]`
- `[[60], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path openai/gpt-oss-120b --tp 8
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
