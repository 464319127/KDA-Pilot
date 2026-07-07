# Profile evidence — step35_flash__void_moe_top_k_256_float_const_b

**Standalone kernel target: 3.3% of total serving GPU time** (max across scenarios) on
`stepfun-ai/Step-3.5-Flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `stepfun-ai/Step-3.5-Flash` (slug `step35_flash`, tp=4)
- Python interface: `<confirm via capture; profiler family=void_moe_top_k_256_float_const_b>`
- Kernel family: `void_moe_top_k_256_float_const_b`  ·  Category: `moe`
- GPU kernel(s): `void moeTopK<256>(float const*, bool const*, float*, int*, int, int, int, int, bool, float`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.79% |
| random | conc 32 | 3.09% |
| random | conc 100 | 2.85% |
| sharegpt | conc 1 | 3.00% |
| sharegpt | conc 32 | 2.88% |
| sharegpt | conc 100 | 3.28% |

**Peak: 3.3% in `sharegpt_high` (sharegpt, concurrency 100).**

## Input shapes (profiler)
- `[[1603073], []]`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1], []]`
- `[[1]]`
- `[[[0], [38]], []]`
- `[[[1]], [], [], [], [], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path stepfun-ai/Step-3.5-Flash --tp 4 --trust-remote-code --reasoning-parser step3p5
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
