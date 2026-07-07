# Profile evidence — minimax_m25__rmsnorm

**Standalone kernel target: 3.0% of total serving GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M2.5`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `MiniMaxAI/MiniMax-M2.5` (slug `minimax_m25`, tp=8)
- Python interface: `<confirm via capture; profiler family=rmsnorm>`
- Kernel family: `rmsnorm`  ·  Category: `norm`
- GPU kernel(s): `void flashinfer::norm::FusedAddRMSNormKernel<8u, __half>(__half*, __half*, __half*, unsign`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.44% |
| random | conc 32 | 2.95% |
| random | conc 100 | 2.54% |
| sharegpt | conc 1 | 2.07% |
| sharegpt | conc 32 | 3.00% |
| sharegpt | conc 100 | 2.19% |

**Peak: 3.0% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[48, 1, 128], [], [], []]`
- `[[64859, 1, 64, 128], [], [], []]`
- `[[64859, 64, 1, 128], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path MiniMaxAI/MiniMax-M2.5 --tp 8 --ep 8 --reasoning-parser minimax-append-think
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
