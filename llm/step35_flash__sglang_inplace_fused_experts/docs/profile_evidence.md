# Profile evidence — step35_flash__sglang_inplace_fused_experts

**Standalone kernel target: 16.1% of total serving GPU time** (max across scenarios) on
`stepfun-ai/Step-3.5-Flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

> Triton MoE expert-GEMM (sglang's own fused_experts/fused_moe kernel) — single-GPU optimizable; NOT the comm-fused trtllm MoE path (excluded).

- Model: `stepfun-ai/Step-3.5-Flash` (slug `step35_flash`, tp=4)
- Python interface: `sglang.inplace_fused_experts`
- Kernel family: `fused_moe_triton`  ·  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 14.66% |
| random | conc 32 | 16.13% |
| random | conc 100 | 14.89% |
| sharegpt | conc 1 | 13.63% |
| sharegpt | conc 32 | 13.10% |
| sharegpt | conc 100 | 15.80% |

**Peak: 16.1% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[16, 4096], [288, 640, 4096], [288, 4096, 320], [16, 8], [16, 8], [], [], [], [`
- `[[38, 4096], [288, 640, 4096], [288, 4096, 320], [38, 8], [38, 8], [], [], [], [`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path stepfun-ai/Step-3.5-Flash --tp 4 --trust-remote-code --reasoning-parser step3p5
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
