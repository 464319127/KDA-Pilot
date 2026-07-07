# Profile evidence — kimi_linear__sglang_inplace_fused_experts

**Standalone kernel target: 26.4% of total serving GPU time** (max across scenarios) on
`moonshotai/Kimi-Linear-48B-A3B-Instruct`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

> Triton MoE expert-GEMM (sglang's own fused_experts/fused_moe kernel) — single-GPU optimizable; NOT the comm-fused trtllm MoE path (excluded).

- Model: `moonshotai/Kimi-Linear-48B-A3B-Instruct` (slug `kimi_linear`, tp=4)
- Python interface: `sglang.inplace_fused_experts`
- Kernel family: `fused_moe_triton`  ·  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 14.79% |
| random | conc 100 | 25.03% |
| sharegpt | conc 1 | 4.45% |
| sharegpt | conc 32 | 10.04% |
| sharegpt | conc 100 | 26.35% |

**Peak: 26.4% in `sharegpt_high` (sharegpt, concurrency 100).**

## Input shapes (profiler)
- `[[1024, 15], [], [], [], [], []]`
- `[[131072, 512], []]`
- `[[15, 8, 128], [], [], []]`
- `[[16384, 2304], [256, 512, 2304], [256, 2304, 256], [16384, 8], [16384, 8], [], `
- `[[16384, 8, 128], [], [], []]`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[31], [], []]`
- `[[48, 3072], [1, 48, 8, 128], [1, 48, 8], [1, 48, 8, 128], []]`
- `[[55], [], []]`
- `[[9967, 2304], [256, 512, 2304], [256, 2304, 256], [9967, 8], [9967, 8], [], [],`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct --tp 4 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
