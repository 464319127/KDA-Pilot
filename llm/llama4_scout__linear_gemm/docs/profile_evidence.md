# Profile evidence — llama4_scout__linear_gemm

**e2e-optimization target: 7.3% of total GPU time** (max across scenarios) on
`meta-llama/Llama-4-Scout-17B-16E-Instruct`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `meta-llama/Llama-4-Scout-17B-16E-Instruct` (slug `llama4_scout`, tp=8)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_128x256_64x6_2x2_2cta_h_bz_TNT`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_splitK_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.30% |
| random | conc 32 | 2.48% |
| sharegpt | conc 1 | 7.26% |

**Peak: 7.3% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1], []]`
- `[[9795, 5120], [5120, 2048]]`
- `[[[704], [1]], []]`
- `[[], [], [], [], [], []]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8 --trust-remote-code --mem-fraction-static 0.8 --context-length 65536
```
After optimizing, re-run **sharegpt_low** to validate the e2e effect.
