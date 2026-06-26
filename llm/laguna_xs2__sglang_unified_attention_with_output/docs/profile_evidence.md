# Profile evidence — laguna_xs2__sglang_unified_attention_with_output

**e2e-optimization target: 11.3% of total GPU time** (max across scenarios) on
`poolside/Laguna-XS.2-NVFP4`, from the exact cookbook-aligned profile. Clean Python interface (profiler provenance).

- Model: `poolside/Laguna-XS.2-NVFP4` (slug `laguna_xs2`, tp=1)
- Python interface: `sglang.unified_attention_with_output`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvSlidingOrChunkedCausalP64MultiCtasKvCgaVar`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvSlidingOrChunkedCausalP64VarSeqQ128Kv128Pe`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvSlidingOrChunkedCausalP64VarSeqQ8Kv128Pers`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.72% |
| random | conc 32 | 4.78% |
| random | conc 100 | 11.35% |
| sharegpt | conc 1 | 3.07% |
| sharegpt | conc 32 | 9.11% |
| sharegpt | conc 100 | 5.78% |

**Peak: 11.3% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[1], [], []]`
- `[[1]]`
- `[[2049, 262148], []]`
- `[[29], [29], []]`
- `[[352, 8192], [352, 8, 128], [352, 8, 128], [352, 8192], [], [], [], [], [], [],`
- `[[5120, 8192], [5120, 8, 128], [5120, 8, 128], [5120, 8192], [], [], [], [], [],`
- `[[5632, 6144], [5632, 8, 128], [5632, 8, 128], [5632, 6144], [], [], [], [], [],`
- `[[5632, 8192], [5632, 8, 128], [5632, 8, 128], [5632, 8192], [], [], [], [], [],`
- `[[576, 6144], [576, 8, 128], [576, 8, 128], [576, 6144], [], [], [], [], [], [],`
- `[[576, 8192], [576, 8, 128], [576, 8, 128], [576, 8192], [], [], [], [], [], [],`
- `[[830], [], [], [], []]`
- `[[[768], [43]], []]`

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path poolside/Laguna-XS.2-NVFP4 --tp 1 --trust-remote-code
```
After optimizing, re-run **random_high** to validate the e2e effect.
