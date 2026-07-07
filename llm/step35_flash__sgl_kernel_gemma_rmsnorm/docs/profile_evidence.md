# Profile evidence — step35_flash__sgl_kernel_gemma_rmsnorm

**Standalone kernel target: 3.1% of total serving GPU time** (max across scenarios) on
`stepfun-ai/Step-3.5-Flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `stepfun-ai/Step-3.5-Flash` (slug `step35_flash`, tp=4)
- Python interface: `sgl_kernel.gemma_rmsnorm`
- Kernel family: `rmsnorm`  ·  Category: `norm`
- GPU kernel(s): `void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16*, __`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.64% |
| random | conc 32 | 2.92% |
| random | conc 100 | 2.69% |
| sharegpt | conc 1 | 2.82% |
| sharegpt | conc 32 | 2.71% |
| sharegpt | conc 100 | 3.08% |

**Peak: 3.1% in `sharegpt_high` (sharegpt, concurrency 100).**

## Input shapes (profiler)
- `[[16, 4096], [16, 4096], [4096], [], []]`
- `[[1], [], [], [], []]`
- `[[1], [], []]`
- `[[1], []]`
- `[[1]]`
- `[[38, 4096], [38, 4096], [4096], [], []]`
- `[[[1]], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path stepfun-ai/Step-3.5-Flash --tp 4 --trust-remote-code --reasoning-parser step3p5
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
