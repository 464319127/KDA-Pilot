# Profile evidence — kimi_linear__linear_gemm

**Standalone kernel target: 16.5% of total serving GPU time** (max across scenarios) on
`moonshotai/Kimi-Linear-48B-A3B-Instruct`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `moonshotai/Kimi-Linear-48B-A3B-Instruct` (slug `kimi_linear`, tp=4)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_16x64_64x16_2x1_v_bz_TNN`, `nvjet_sm100_tst_16x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 16.53% |
| random | conc 100 | 2.88% |
| sharegpt | conc 1 | 4.23% |

**Peak: 16.5% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[1, 163840], [], [], [], []]`
- `[[1, 163840], [], [], []]`
- `[[1, 163840], [], []]`
- `[[16384, 2, 128], [], [], []]`
- `[[1], [1], []]`
- `[[1]]`
- `[[2], [2], []]`
- `[[3577], [], [], [], []]`
- `[[47], [], [], []]`
- `[[80], [], [], []]`
- `[[[1]], [], [], [], [], []]`
- `[[], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct --tp 4 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
