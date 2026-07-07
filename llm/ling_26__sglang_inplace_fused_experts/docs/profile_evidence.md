# Profile evidence — ling_26__sglang_inplace_fused_experts

**Standalone kernel target: 32.3% of total serving GPU time** (max across scenarios) on
`inclusionAI/Ling-2.6-flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

> Triton MoE expert-GEMM (sglang's own fused_experts/fused_moe kernel) — single-GPU optimizable; NOT the comm-fused trtllm MoE path (excluded).

- Model: `inclusionAI/Ling-2.6-flash` (slug `ling_26`, tp=4)
- Python interface: `sglang.inplace_fused_experts`
- Kernel family: `fused_moe_triton`  ·  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 9.39% |
| random | conc 32 | 32.32% |
| random | conc 100 | 18.60% |
| sharegpt | conc 1 | 9.27% |
| sharegpt | conc 32 | 26.15% |
| sharegpt | conc 100 | 18.55% |

**Peak: 32.3% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[12199, 4096], [256, 512, 4096], [256, 4096, 256], [12199, 8], [12199, 8], [], `
- `[[3215, 4096], [256, 512, 4096], [256, 4096, 256], [3215, 8], [3215, 8], [], [],`
- `[[334, 4096], [256, 512, 4096], [256, 4096, 256], [334, 8], [334, 8], [], [], []`
- `[[39, 4096], [256, 512, 4096], [256, 4096, 256], [39, 8], [39, 8], [], [], [], [`
- `[[39, 8, 4096], [39, 4096], []]`
- `[[44, 4096], [256, 512, 4096], [256, 4096, 256], [44, 8], [44, 8], [], [], [], [`
- `[[64]]`
- `[[9780, 4096], [256, 512, 4096], [256, 4096, 256], [9780, 8], [9780, 8], [], [],`
- `[[], [], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path inclusionAI/Ling-2.6-flash --tp 4 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
