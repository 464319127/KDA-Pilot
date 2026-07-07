# Profile evidence — mimo_v25__fused_add_rmsnorm

**Standalone kernel target: 34.6% of total serving GPU time** (max across scenarios) on
`XiaomiMiMo/MiMo-V2.5`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `XiaomiMiMo/MiMo-V2.5` (slug `mimo_v25`, tp=4)
- Python interface: `<confirm via capture; profiler family=fused_add_rmsnorm>`
- Kernel family: `fused_add_rmsnorm`  ·  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 31.79% |
| random | conc 32 | 16.86% |
| random | conc 100 | 23.71% |
| sharegpt | conc 1 | 34.62% |
| sharegpt | conc 32 | 14.71% |
| sharegpt | conc 100 | 25.82% |

**Peak: 34.6% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[103, 3712], []]`
- `[[103, 4096], [103, 4096], [103, 8], [], [], [], [], [], [], []]`
- `[[103, 4096], [], []]`
- `[[196, 3072], []]`
- `[[196, 4096], [], []]`
- `[[86, 256], [256], [86, 4096], [32, 86], [256, 1024, 4096], [256, 8, 32], [256, `
- `[[89, 256], [256], [89, 4096], [32, 89], [256, 1024, 4096], [256, 8, 32], [256, `
- `[[89, 256], []]`
- `[[89, 4096], [4096, 256]]`
- `[[89, 4096], [89, 4096], []]`
- `[[89, 4096], [], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path XiaomiMiMo/MiMo-V2.5 --tp 4 --trust-remote-code --attention-backend fa4 --mm-attention-backend fa4 --moe-runner-backend flashinfer_trtllm --mem-fraction-static 0.65 --chunked-prefill-size 16384
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
