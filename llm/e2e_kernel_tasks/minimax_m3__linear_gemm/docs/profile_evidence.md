# Profile evidence — minimax_m3__linear_gemm

**e2e-optimization target: 24.1% of total GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M3-MXFP8`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `MiniMaxAI/MiniMax-M3-MXFP8` (slug `minimax_m3`, tp=8)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `post_reorder_deepgemm_triton_kernel`, `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 3`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 14.37% |
| random | conc 32 | 23.04% |
| random | conc 100 | 24.08% |
| sharegpt | conc 1 | 14.45% |
| sharegpt | conc 32 | 19.32% |
| sharegpt | conc 100 | 15.61% |

**Peak: 24.1% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[1, 3597, 8], []]`
- `[[151], [], [], [], []]`
- `[[15], [15], []]`
- `[[15], []]`
- `[[16, 1, 128], [], [], [], []]`
- `[[1792, 8, 128], [1792, 1, 128], [1792, 1, 128], [1792, 1024], [1792, 128], [179`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1]]`
- `[[3072, 8, 128], [3072, 1, 128], [3072, 1, 128], [3072, 1024], [3072, 128], [307`
- `[[33], [], [], [], [], []]`
- `[[33], []]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path MiniMaxAI/MiniMax-M3-MXFP8 --tp 8 --trust-remote-code --quantization mxfp8 --attention-backend fa4 --page-size 128 --mem-fraction-static 0.65
```
After optimizing, re-run **random_high** to validate the e2e effect.
