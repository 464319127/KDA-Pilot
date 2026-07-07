# Profile evidence — minimax_m27__rmsnorm

**Standalone kernel target: 3.9% of total serving GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M2.7`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `MiniMaxAI/MiniMax-M2.7` (slug `minimax_m27`, tp=8)
- Python interface: `<confirm via capture; profiler family=rmsnorm>`
- Kernel family: `rmsnorm`  ·  Category: `norm`
- GPU kernel(s): `void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, __nv_bfloa`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 3.90% |
| random | conc 32 | 3.32% |
| random | conc 100 | 2.40% |
| sharegpt | conc 1 | 2.33% |
| sharegpt | conc 32 | 3.28% |
| sharegpt | conc 100 | 2.40% |

**Peak: 3.9% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[1], [], [], [], []]`
- `[[1]]`
- `[[59, 1, 128], []]`
- `[[64, 1, 128], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path MiniMaxAI/MiniMax-M2.7 --tp 8 --ep 8 --tool-call-parser minimax-m2 --reasoning-parser minimax-append-think
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
