# Profile evidence — glm_46__fused_moe_triton

**Standalone kernel target: 34.8% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-4.6-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

> Triton MoE expert-GEMM (sglang's own fused_experts/fused_moe kernel) — single-GPU optimizable; NOT the comm-fused trtllm MoE path (excluded).

- Model: `zai-org/GLM-4.6-FP8` (slug `glm_46`, tp=8)
- Python interface: `<confirm via capture; profiler family=fused_moe_triton>`
- Kernel family: `fused_moe_triton`  ·  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 19.81% |
| random | conc 32 | 32.49% |
| random | conc 100 | 34.80% |
| sharegpt | conc 1 | 18.43% |
| sharegpt | conc 32 | 28.40% |
| sharegpt | conc 100 | 32.75% |

**Peak: 34.8% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[112, 1536], [112, 1, 128], [112, 1, 128], [112, 1536], [], [], [], [], [], [],`
- `[[112, 1536], [], [], []]`
- `[[1536, 1, 128], [], [], []]`
- `[[1], []]`
- `[[2193, 12, 128]]`
- `[[2816, 1, 128], [], [], []]`
- `[[325], [], [], []]`
- `[[32], [32], []]`
- `[[80, 12, 128], [19176, 128, 1, 128], [19176, 128, 1, 128], [], [2], [], [], [1]`
- `[[80, 1536], [80, 1536], []]`
- `[[8125, 12, 128], [19176, 128, 1, 128], [19176, 128, 1, 128], [], [27], [], [], `

## Original serving capture command (provenance only)
```bash
sglang serve --model-path zai-org/GLM-4.6-FP8 --tp 8 --reasoning-parser glm45 --tool-call-parser glm45 --attention-backend fa4
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
