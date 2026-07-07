# Profile evidence — qwen3_coder_next__linear_gemm

**Standalone kernel target: 40.5% of total serving GPU time** (max across scenarios) on
`Qwen/Qwen3-Coder-Next`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `Qwen/Qwen3-Coder-Next` (slug `qwen3_coder_next`, tp=2)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_128x24_64x11_4x2_h_bz_TNT`, `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_16x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x16_64x16_1x2_h_bz_TNT`, `nvjet_sm100_tst_64x24_64x16_1x2_h_bz_TNT`, `nvjet_sm100_tst_64x24_64x16_4x1_v_bz_TNT`, `nvjet_sm100_tst_64x64_64x16_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`, `nvjet_sm100_tst_8x64_64x16_4x1_v_bz_TNN`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 40.52% |
| random | conc 32 | 12.37% |
| random | conc 100 | 15.02% |
| sharegpt | conc 1 | 27.03% |
| sharegpt | conc 32 | 18.76% |
| sharegpt | conc 100 | 16.59% |

**Peak: 40.5% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[1, 118, 16, 128], []]`
- `[[1, 151936], [], [], []]`
- `[[1, 151936], [], []]`
- `[[118, 2048], [2048, 512]]`
- `[[118, 2048], [], [], [], [], []]`
- `[[16384, 2048], [2048, 6144]]`
- `[[16384, 4096], [], [], []]`
- `[[17, 2048], [], [], [], [], []]`
- `[[17, 2048], []]`
- `[[17184, 128], []]`
- `[[1], [1], []]`
- `[[1], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path Qwen/Qwen3-Coder-Next --tp 2 --tool-call-parser qwen3_coder
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
