# Profile evidence — ministral3_14b__quant_fp8

**Standalone kernel target: 6.7% of total serving GPU time** (max across scenarios) on
`mistralai/Ministral-3-14B-Instruct-2512`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `mistralai/Ministral-3-14B-Instruct-2512` (slug `ministral3_14b`, tp=1)
- Python interface: `<confirm via capture; profiler family=quant_fp8>`
- Kernel family: `quant_fp8`  ·  Category: `quant_gemm`
- GPU kernel(s): `_static_quant_fp8`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 6.68% |
| random | conc 32 | 5.36% |
| random | conc 100 | 4.72% |
| sharegpt | conc 1 | 6.61% |
| sharegpt | conc 32 | 5.08% |
| sharegpt | conc 100 | 4.90% |

**Peak: 6.7% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[1228, 5120], []]`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1], []]`
- `[[32], [32], []]`
- `[[42], [], []]`
- `[[], [], [], [], [], []]`
- `[[]]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path mistralai/Ministral-3-14B-Instruct-2512 --tp 1 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
