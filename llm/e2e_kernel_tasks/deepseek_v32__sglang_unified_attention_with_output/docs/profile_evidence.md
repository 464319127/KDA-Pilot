# Profile evidence — deepseek_v32__sglang_unified_attention_with_output

**e2e-optimization target: 13.3% of total GPU time** (max across scenarios) on
`nvidia/DeepSeek-V3.2-NVFP4`, from the exact cookbook-aligned profile. Clean Python interface (profiler provenance).

- Model: `nvidia/DeepSeek-V3.2-NVFP4` (slug `deepseek_v32`, tp=4)
- Python interface: `sglang.unified_attention_with_output`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk192HV128SeparateQkvCausalVarSeqQ128Kv128Persistent`, `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1Mult`, `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1VarSeqQ32Kv128P`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.59% |
| random | conc 100 | 5.65% |
| sharegpt | conc 1 | 2.82% |
| sharegpt | conc 32 | 13.29% |
| sharegpt | conc 100 | 5.09% |

**Peak: 13.3% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[1, 129280], [], [], []]`
- `[[13285, 7168], [7168, 128]]`
- `[[13285, 7168], [7168, 2112]]`
- `[[1], [1], []]`
- `[[1], [], [], [], [], []]`
- `[[2310, 2048], [2048, 7168], [2432, 256], [256, 7168], [], [], []]`
- `[[66]]`
- `[[960, 32, 512], [960, 1, 512], [960, 1, 512], [960, 16384], [], [], [960, 32, 6`
- `[[[64], [39]], []]`
- `[[], [], [], [], [], []]`
- `[[], [], [], []]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model nvidia/DeepSeek-V3.2-NVFP4 --tp 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3
```
After optimizing, re-run **sharegpt_mid** to validate the e2e effect.
