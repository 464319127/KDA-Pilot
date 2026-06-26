# Profile evidence — ring_26_1t__sglang_inplace_fused_experts

**e2e-optimization target: 31.2% of total GPU time** (max across scenarios) on
`inclusionAI/Ring-2.6-1T`, from the exact cookbook-aligned profile. Clean Python interface (profiler provenance).

> Triton MoE expert-GEMM (sglang's own fused_experts/fused_moe kernel) — single-GPU optimizable; NOT the comm-fused trtllm MoE path (excluded).

- Model: `inclusionAI/Ring-2.6-1T` (slug `ring_26_1t`, tp=8)
- Python interface: `sglang.inplace_fused_experts`
- Kernel family: `fused_moe_triton`  ·  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 12.39% |
| random | conc 32 | 31.18% |
| random | conc 100 | 30.80% |
| sharegpt | conc 1 | 12.58% |
| sharegpt | conc 32 | 28.59% |
| sharegpt | conc 100 | 28.93% |

**Peak: 31.2% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[14375, 3072], []]`
- `[[14375, 8192], [256, 512, 8192], [256, 8192, 256], [14375, 8], [14375, 8], [], `
- `[[16384, 8, 128], [], [], [], [], []]`
- `[[16384, 8192], [256, 512, 8192], [256, 8192, 256], [16384, 8], [16384, 8], [], `
- `[[39, 8192], [256, 512, 8192], [256, 8192, 256], [39, 8], [39, 8], [], [], [], [`
- `[[44, 8192], [256, 512, 8192], [256, 8192, 256], [44, 8], [44, 8], [], [], [], [`
- `[[9780, 1024], []]`
- `[[9780, 8192], [256, 512, 8192], [256, 8192, 256], [9780, 8], [9780, 8], [], [],`

## Reproduce (cookbook-aligned)
```bash
sglang serve --model-path inclusionAI/Ring-2.6-1T --tp-size 8 --trust-remote-code --mem-fraction-static 0.8 --tool-call-parser glm --reasoning-parser deepseek-r1
```
After optimizing, re-run **random_mid** to validate the e2e effect.
