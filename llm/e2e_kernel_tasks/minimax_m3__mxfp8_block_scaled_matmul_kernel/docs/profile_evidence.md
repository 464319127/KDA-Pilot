# Profile evidence — minimax_m3__mxfp8_block_scaled_matmul_kernel

**e2e-optimization target: 10.2% of total GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M3-MXFP8`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `MiniMaxAI/MiniMax-M3-MXFP8` (slug `minimax_m3`, tp=8)
- Python interface: `<confirm via capture; profiler family=mxfp8_block_scaled_matmul_kernel>`
- Kernel family: `mxfp8_block_scaled_matmul_kernel`  ·  Category: `quant_gemm`
- GPU kernel(s): `_mxfp8_block_scaled_matmul_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 9.77% |
| random | conc 32 | 5.89% |
| random | conc 100 | 5.61% |
| sharegpt | conc 1 | 10.17% |
| sharegpt | conc 32 | 5.39% |
| sharegpt | conc 100 | 4.00% |

**Peak: 10.2% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[151], [], [], [], []]`
- `[[1]]`
- `[[3072, 1024], [3072, 1, 128], [3072, 1, 128], [3072, 1024], [], [], [], [], [],`
- `[[4096], [], [], []]`
- `[[], [], [], [], [], []]`
- `[[]]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path MiniMaxAI/MiniMax-M3-MXFP8 --tp 8 --trust-remote-code --quantization mxfp8 --attention-backend fa4 --page-size 128 --mem-fraction-static 0.65
```
After optimizing, re-run **sharegpt_low** to validate the e2e effect.
