# Profile evidence — mimo_v25__rmsnorm

**e2e-optimization target: 3.9% of total GPU time** (max across scenarios) on
`XiaomiMiMo/MiMo-V2.5`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `XiaomiMiMo/MiMo-V2.5` (slug `mimo_v25`, tp=4)
- Python interface: `<confirm via capture; profiler family=rmsnorm>`
- Kernel family: `rmsnorm`  ·  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 3.85% |

**Peak: 3.9% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[], [], [], [], [], []]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path XiaomiMiMo/MiMo-V2.5 --tp 4 --trust-remote-code --attention-backend fa4 --mm-attention-backend fa4 --moe-runner-backend flashinfer_trtllm --mem-fraction-static 0.65 --chunked-prefill-size 16384
```
After optimizing, re-run **random_low** to validate the e2e effect.
