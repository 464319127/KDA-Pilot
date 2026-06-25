# Profile evidence — deepseek_v32__quant_fp8

**e2e-optimization target: 3.2% of total GPU time** (max across scenarios) on
`nvidia/DeepSeek-V3.2-NVFP4`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/DeepSeek-V3.2-NVFP4` (slug `deepseek_v32`, tp=4)
- Python interface: `<confirm via capture; profiler family=quant_fp8>`
- Kernel family: `quant_fp8`  ·  Category: `quant_gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeSwizzledKern`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 3.18% |
| sharegpt | conc 1 | 3.14% |

**Peak: 3.2% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[18], [], [], [], []]`
- `[[39, 1, 32, 512], [39, 1, 32, 512], []]`
- `[[], [48, 128], [48, 64, 128], [48, 64, 1], [48, 2048]]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model nvidia/DeepSeek-V3.2-NVFP4 --tp 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3
```
After optimizing, re-run **random_low** to validate the e2e effect.
