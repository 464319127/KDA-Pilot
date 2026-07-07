# Profile evidence — minimax_m2__per_token_group_quant

**Standalone kernel target: 4.7% of total serving GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M2`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `MiniMaxAI/MiniMax-M2` (slug `minimax_m2`, tp=4)
- Python interface: `<confirm via capture; profiler family=per_token_group_quant>`
- Kernel family: `per_token_group_quant`  ·  Category: `quant_gemm`
- GPU kernel(s): `void (anonymous namespace)::per_token_group_quant_8bit_v2_kernel<(anonymous namespace)::Na`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 4.72% |
| random | conc 32 | 2.21% |
| sharegpt | conc 1 | 4.65% |
| sharegpt | conc 32 | 2.09% |

**Peak: 4.7% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[1, 200064], [], []]`
- `[[1]]`
- `[[25357, 64, 2, 128], [], [], []]`
- `[[9468, 2, 128], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path MiniMaxAI/MiniMax-M2 --tp 4 --reasoning-parser minimax-append-think --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
