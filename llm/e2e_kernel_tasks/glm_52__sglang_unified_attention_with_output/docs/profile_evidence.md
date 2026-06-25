# Profile evidence — glm_52__sglang_unified_attention_with_output

**e2e-optimization target: 14.2% of total GPU time** (max across scenarios) on
`zai-org/GLM-5.2-FP8`, from the exact cookbook-aligned profile. Clean Python interface (profiler provenance).

- Model: `zai-org/GLM-5.2-FP8` (slug `glm_52`, tp=8)
- Python interface: `sglang.unified_attention_with_output`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1Mult`, `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1VarSeqQ8Kv128Pe`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.76% |
| random | conc 100 | 4.33% |
| sharegpt | conc 1 | 3.28% |
| sharegpt | conc 32 | 14.21% |
| sharegpt | conc 100 | 4.60% |

**Peak: 14.2% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[1536, 8, 512], [1536, 1, 512], [1536, 1, 512], [1536, 4096], [], [], [1536, 8,`
- `[[1], [1], []]`
- `[[1], [], [], [], []]`
- `[[1], []]`
- `[[2048, 8, 512], [2048, 1, 512], [2048, 1, 512], [2048, 4096], [], [], [2048, 8,`
- `[[9137, 1, 8, 576], []]`
- `[[]]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path zai-org/GLM-5.2-FP8 --tp 8 --trust-remote-code --mem-fraction-static 0.8
```
After optimizing, re-run **sharegpt_mid** to validate the e2e effect.
