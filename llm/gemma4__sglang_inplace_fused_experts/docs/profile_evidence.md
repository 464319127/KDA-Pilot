# Profile evidence — gemma4__sglang_inplace_fused_experts

**Standalone kernel target: 50.8% of total serving GPU time** (max across scenarios) on
`google/gemma-4-26B-A4B-it`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

> Triton MoE expert-GEMM (sglang's own fused_experts/fused_moe kernel) — single-GPU optimizable; NOT the comm-fused trtllm MoE path (excluded).

- Model: `google/gemma-4-26B-A4B-it` (slug `gemma4`, tp=1)
- Python interface: `sglang.inplace_fused_experts`
- Kernel family: `fused_moe_triton`  ·  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 22.87% |
| random | conc 32 | 45.23% |
| random | conc 100 | 50.81% |
| sharegpt | conc 1 | 21.68% |
| sharegpt | conc 32 | 43.37% |
| sharegpt | conc 100 | 50.02% |

**Peak: 50.8% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[11254, 2816], [128, 1408, 2816], [128, 2816, 704], [11254, 8], [11254, 8], [],`
- `[[17, 2816], [128, 1408, 2816], [128, 2816, 704], [17, 8], [17, 8], [], [], [], `
- `[[1785, 2816], [128, 1408, 2816], [128, 2816, 704], [1785, 8], [1785, 8], [], []`
- `[[1902, 2816], [128, 1408, 2816], [128, 2816, 704], [1902, 8], [1902, 8], [], []`
- `[[38, 2816], [128, 1408, 2816], [128, 2816, 704], [38, 8], [38, 8], [], [], [], `
- `[[7081, 2816], [128, 1408, 2816], [128, 2816, 704], [7081, 8], [7081, 8], [], []`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path google/gemma-4-26B-A4B-it --reasoning-parser gemma4 --tool-call-parser gemma4
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
