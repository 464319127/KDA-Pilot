# Profile evidence — minimax_m2__fused_moe_triton

**Standalone kernel target: 48.1% of total serving GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M2`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

> Triton MoE expert-GEMM (sglang's own fused_experts/fused_moe kernel) — single-GPU optimizable; NOT the comm-fused trtllm MoE path (excluded).

- Model: `MiniMaxAI/MiniMax-M2` (slug `minimax_m2`, tp=4)
- Python interface: `<confirm via capture; profiler family=fused_moe_triton>`
- Kernel family: `fused_moe_triton`  ·  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 23.61% |
| random | conc 32 | 43.66% |
| random | conc 100 | 48.15% |
| sharegpt | conc 1 | 24.78% |
| sharegpt | conc 32 | 43.36% |
| sharegpt | conc 100 | 47.91% |

**Peak: 48.1% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[25357, 64, 2, 128], [], [], []]`
- `[[256], [], [], [], []]`
- `[[2797, 2, 128], []]`
- `[[9468, 2, 128], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path MiniMaxAI/MiniMax-M2 --tp 4 --reasoning-parser minimax-append-think --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
