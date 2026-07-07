# Profile evidence — laguna_xs2__sglang_unified_attention_with_output

**Standalone kernel target: 11.3% of total serving GPU time** (max across scenarios) on
`poolside/Laguna-XS.2-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

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

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path poolside/Laguna-XS.2-NVFP4 --tp 1 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
