# Profile evidence — minimax_m27__fused_moe_triton

**Standalone kernel target: 32.5% of total serving GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M2.7`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

> Triton MoE expert-GEMM (sglang's own fused_experts/fused_moe kernel) — single-GPU optimizable; NOT the comm-fused trtllm MoE path (excluded).

- Model: `MiniMaxAI/MiniMax-M2.7` (slug `minimax_m27`, tp=8)
- Python interface: `<confirm via capture; profiler family=fused_moe_triton>`
- Kernel family: `fused_moe_triton`  ·  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 16.64% |
| random | conc 32 | 31.40% |
| random | conc 100 | 32.52% |
| sharegpt | conc 1 | 18.24% |
| sharegpt | conc 32 | 30.62% |
| sharegpt | conc 100 | 32.22% |

**Peak: 32.5% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[19]]`
- `[[1], [], [], [], []]`
- `[[5632, 768], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path MiniMaxAI/MiniMax-M2.7 --tp 8 --ep 8 --tool-call-parser minimax-m2 --reasoning-parser minimax-append-think
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
