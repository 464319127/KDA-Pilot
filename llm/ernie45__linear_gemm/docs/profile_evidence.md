# Profile evidence — ernie45__linear_gemm

**Standalone kernel target: 29.1% of total serving GPU time** (max across scenarios) on
`baidu/ERNIE-4.5-21B-A3B-PT`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `baidu/ERNIE-4.5-21B-A3B-PT` (slug `ernie45`, tp=1)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_128x256_64x6_2x2_2cta_h_bz_TNT`, `nvjet_sm100_tst_32x64_64x16_2x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_64x64_64x16_2x2_2cta_h_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_2x1_v_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 27.31% |
| random | conc 32 | 13.54% |
| random | conc 100 | 2.90% |
| sharegpt | conc 1 | 29.07% |
| sharegpt | conc 32 | 9.63% |
| sharegpt | conc 100 | 2.55% |

**Peak: 29.1% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[1, 1], [1, 1], []]`
- `[[1, 2560], [2560, 103424]]`
- `[[100, 103424], [], []]`
- `[[100]]`
- `[[104, 17], [104, 17], []]`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1]]`
- `[[256], [256], []]`
- `[[256], [], [], [], [], [], []]`
- `[[284], [], [], []]`
- `[[32, 16], [32, 16], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path baidu/ERNIE-4.5-21B-A3B-PT --tp 1
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
