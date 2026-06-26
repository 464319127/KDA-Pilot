# Profile evidence — laguna_xs2__void_cvt_fp16_to_fp4_nv_bfloat16

**e2e-optimization target: 9.2% of total GPU time** (max across scenarios) on
`poolside/Laguna-XS.2-NVFP4`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `poolside/Laguna-XS.2-NVFP4` (slug `laguna_xs2`, tp=1)
- Python interface: `<confirm via capture; profiler family=void_cvt_fp16_to_fp4_nv_bfloat16>`
- Kernel family: `void_cvt_fp16_to_fp4_nv_bfloat16`  ·  Category: `quant_gemm`
- GPU kernel(s): `void cvt_fp16_to_fp4<__nv_bfloat16, false, false>(int, int, __nv_bfloat16 const*, float co`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 9.17% |
| random | conc 100 | 6.27% |
| sharegpt | conc 100 | 6.06% |

**Peak: 9.2% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[0], [0], []]`
- `[[12663, 64, 8, 128], [], [], []]`
- `[[12663, 64, 8, 128], []]`
- `[[13, 100352], [], []]`
- `[[144, 8192], [144, 8, 128], [144, 8, 128], [144, 8192], [], [], [], [], [], [],`
- `[[1], [1], []]`
- `[[316], [], [], []]`
- `[[32], [32], []]`
- `[[534, 8192], [534, 8192], []]`
- `[[64], [], [], []]`
- `[[[1280], [64]], []]`
- `[[[384], [2]], []]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path poolside/Laguna-XS.2-NVFP4 --tp 1 --trust-remote-code
```
After optimizing, re-run **random_mid** to validate the e2e effect.
