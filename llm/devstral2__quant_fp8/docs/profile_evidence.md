# Profile evidence — devstral2__quant_fp8

**Standalone kernel target: 6.6% of total serving GPU time** (max across scenarios) on
`mistralai/Devstral-2-123B-Instruct-2512`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `mistralai/Devstral-2-123B-Instruct-2512` (slug `devstral2`, tp=8)
- Python interface: `<confirm via capture; profiler family=quant_fp8>`
- Kernel family: `quant_fp8`  ·  Category: `quant_gemm`
- GPU kernel(s): `_static_quant_fp8`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 6.60% |
| random | conc 32 | 4.50% |
| random | conc 100 | 4.08% |
| sharegpt | conc 1 | 6.58% |
| sharegpt | conc 32 | 4.52% |
| sharegpt | conc 100 | 4.17% |

**Peak: 6.6% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[1392, 1536], [1392, 1536], []]`
- `[[1536, 1536], [1536, 1, 128], [1536, 1, 128], [1536, 1536], [], [], [], [], [],`
- `[[192], [], [], [], []]`
- `[[1], []]`
- `[[1]]`
- `[[734], [], [], []]`
- `[[7807, 1536], [7807, 1536], []]`
- `[[8192, 1536], [], [], [], []]`
- `[[8704, 1536], [], [], [], []]`
- `[[[1], [39375]], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path mistralai/Devstral-2-123B-Instruct-2512 --tp 8 --reasoning-parser mistral --tool-call-parser mistral
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
