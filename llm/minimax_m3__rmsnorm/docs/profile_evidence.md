# Profile evidence — minimax_m3__rmsnorm

**Standalone kernel target: 20.0% of total serving GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M3-MXFP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `MiniMaxAI/MiniMax-M3-MXFP8` (slug `minimax_m3`, tp=8)
- Python interface: `<confirm via capture; profiler family=rmsnorm>`
- Kernel family: `rmsnorm`  ·  Category: `norm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm`, `void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, __nv_bfloa`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 19.96% |
| random | conc 32 | 10.43% |
| random | conc 100 | 12.05% |
| sharegpt | conc 1 | 19.62% |
| sharegpt | conc 32 | 10.34% |
| sharegpt | conc 100 | 13.32% |

**Peak: 20.0% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[10, 128], []]`
- `[[10, 128]]`
- `[[16, 8, 128], [16, 1, 128], [16, 1, 128], [16, 1024], [16, 128], [16, 1, 128], `
- `[[1]]`
- `[[48, 8, 128], [48, 1, 128], [48, 1, 128], [48, 1024], [48, 128], [48, 1, 128], `
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path MiniMaxAI/MiniMax-M3-MXFP8 --tp 8 --trust-remote-code --quantization mxfp8 --attention-backend fa4 --page-size 128 --mem-fraction-static 0.65
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
