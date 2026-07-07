# Profile evidence — ling_26__fused_add_rmsnorm

**Standalone kernel target: 18.0% of total serving GPU time** (max across scenarios) on
`inclusionAI/Ling-2.6-flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `inclusionAI/Ling-2.6-flash` (slug `ling_26`, tp=4)
- Python interface: `<confirm via capture; profiler family=fused_add_rmsnorm>`
- Kernel family: `fused_add_rmsnorm`  ·  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 100 | 17.97% |

**Peak: 18.0% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[39, 4096], [], []]`
- `[[39, 4096], []]`
- `[[], [39, 512], [39, 256]]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path inclusionAI/Ling-2.6-flash --tp 4 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
