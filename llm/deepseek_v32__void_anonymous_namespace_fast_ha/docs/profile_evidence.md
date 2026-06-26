# Profile evidence — deepseek_v32__void_anonymous_namespace_fast_ha

**e2e-optimization target: 5.1% of total GPU time** (max across scenarios) on
`nvidia/DeepSeek-V3.2-NVFP4`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/DeepSeek-V3.2-NVFP4` (slug `deepseek_v32`, tp=4)
- Python interface: `<confirm via capture; profiler family=void_anonymous_namespace_fast_ha>`
- Kernel family: `void_anonymous_namespace_fast_ha`  ·  Category: `other`
- GPU kernel(s): `void (anonymous namespace)::fast_hadamard_transform_kernel<(anonymous namespace)::FastHada`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| sharegpt | conc 32 | 5.05% |
| sharegpt | conc 100 | 2.20% |

**Peak: 5.1% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[13312, 7168], [], [], [], []]`
- `[[2048, 32, 512], [2048, 1, 512], [2048, 1, 512], [2048, 16384], [], [], [2048, `
- `[[], [2048, 128], [2048, 64, 128], [2048, 64, 1], [2048, 2048]]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model nvidia/DeepSeek-V3.2-NVFP4 --tp 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3
```
After optimizing, re-run **sharegpt_mid** to validate the e2e effect.
