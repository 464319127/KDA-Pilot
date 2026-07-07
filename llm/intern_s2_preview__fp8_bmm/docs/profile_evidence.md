# Profile evidence — intern_s2_preview__fp8_bmm

**Standalone kernel target: 9.0% of total serving GPU time** (max across scenarios) on
`internlm/Intern-S2-Preview`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `internlm/Intern-S2-Preview` (slug `intern_s2_preview`, tp=8)
- Python interface: `<confirm via capture; profiler family=fp8_bmm>`
- Kernel family: `fp8_bmm`  ·  Category: `other`
- GPU kernel(s): `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x64x128_s5_et128x64_m256x64x16_c2x1x1_rM_BN_transOu`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x64x128u2_s5_et128x64_m256x64x16_c2x1x1_rM_BN_trans`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 2.02% |
| random | conc 100 | 8.99% |

**Peak: 9.0% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[[16384, 2048]], [], [], [], [], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path internlm/Intern-S2-Preview --tp 8 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
