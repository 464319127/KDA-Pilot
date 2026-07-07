# Profile evidence — step35_flash__linear_gemm

**Standalone kernel target: 6.9% of total serving GPU time** (max across scenarios) on
`stepfun-ai/Step-3.5-Flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `stepfun-ai/Step-3.5-Flash` (slug `step35_flash`, tp=4)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_tst_32x64_64x16_4x1_v_bz_TNN`, `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.26% |
| random | conc 32 | 5.82% |
| random | conc 100 | 5.37% |
| sharegpt | conc 1 | 6.29% |
| sharegpt | conc 32 | 6.04% |
| sharegpt | conc 100 | 6.87% |

**Peak: 6.9% in `sharegpt_high` (sharegpt, concurrency 100).**

## Input shapes (profiler)
- `[[1, 1], [1, 1], []]`
- `[[16, 4096], [4096, 3584]]`
- `[[1], []]`
- `[[1]]`
- `[[[1]], [], [], [], [], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path stepfun-ai/Step-3.5-Flash --tp 4 --trust-remote-code --reasoning-parser step3p5
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
