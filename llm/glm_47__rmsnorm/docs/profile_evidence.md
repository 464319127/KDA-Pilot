# Profile evidence — glm_47__rmsnorm

**e2e-optimization target: 4.4% of total GPU time** (max across scenarios) on
`nvidia/GLM-4.7-NVFP4`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/GLM-4.7-NVFP4` (slug `glm_47`, tp=8)
- Python interface: `<confirm via capture; profiler family=rmsnorm>`
- Kernel family: `rmsnorm`  ·  Category: `norm`
- GPU kernel(s): `void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, __nv_bfloa`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 4.32% |
| sharegpt | conc 32 | 4.37% |

**Peak: 4.4% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[5672, 1536], []]`
- `[[6586, 1536], [6586, 1536], []]`
- `[[6656, 1536], [6656, 1, 128], [6656, 1, 128], [6656, 1536], [], [], [], [], [],`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model nvidia/GLM-4.7-NVFP4 --tp 8 --quantization modelopt_fp4 --reasoning-parser glm45 --trust-remote-code
```
After optimizing, re-run **sharegpt_mid** to validate the e2e effect.
