# Profile evidence — llama4_scout__fused_add_rmsnorm

**e2e-optimization target: 34.8% of total GPU time** (max across scenarios) on
`meta-llama/Llama-4-Scout-17B-16E-Instruct`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `meta-llama/Llama-4-Scout-17B-16E-Instruct` (slug `llama4_scout`, tp=8)
- Python interface: `<confirm via capture; profiler family=fused_add_rmsnorm>`
- Kernel family: `fused_add_rmsnorm`  ·  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 34.81% |
| random | conc 32 | 21.93% |
| random | conc 100 | 17.87% |
| sharegpt | conc 1 | 32.11% |
| sharegpt | conc 32 | 15.87% |
| sharegpt | conc 100 | 21.40% |

**Peak: 34.8% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[1, 128], []]`
- `[[1, 1], [1, 16], [1, 1], [], []]`
- `[[1, 5, 128], [1, 1, 128], [10485760, 128], [1], [], []]`
- `[[1, 5120], [], []]`
- `[[11, 1], [11, 16], [11, 1], [], [], []]`
- `[[11, 5120], [], []]`
- `[[39, 5120], [5120, 16]]`
- `[[39, 5120], [5120, 896]]`
- `[[39, 5120], [], []]`
- `[[39, 5120]]`
- `[[5120, 640], [], []]`
- `[[5689792, 1, 128], []]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8 --trust-remote-code --mem-fraction-static 0.8 --context-length 65536
```
After optimizing, re-run **random_low** to validate the e2e effect.
