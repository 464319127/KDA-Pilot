# Profile evidence — mistral_small4__attention

**Standalone kernel target: 8.3% of total serving GPU time** (max across scenarios) on
`mistralai/Mistral-Small-4-119B-2603`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `mistralai/Mistral-Small-4-119B-2603` (slug `mistral_small4`, tp=1)
- Python interface: `<confirm via capture; profiler family=attention>`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16H128SeparateQkvCausalVarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk320HV256PagedKvDenseP64MultiCtasKvCgaVarSeqQ16Kv12`, `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk320HV256PagedKvDenseP64MultiCtasKvVarSeqQ16Kv128St`, `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk320HV256PagedKvDenseP64VarSeqQ16Kv128PersistentSwa`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 6.20% |
| random | conc 32 | 2.68% |
| random | conc 100 | 6.13% |
| sharegpt | conc 1 | 6.44% |
| sharegpt | conc 32 | 8.32% |
| sharegpt | conc 100 | 7.03% |

**Peak: 8.3% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[1, 131072], [], [], []]`
- `[[100], [100], []]`
- `[[100]]`
- `[[104, 131072], [], [], []]`
- `[[1738, 32, 128], [1738, 32], [1738, 32, 128], [1738, 32], [1738, 32, 128], [173`
- `[[1738, 4096], []]`
- `[[1], [1], []]`
- `[[1], [], [], []]`
- `[[1], [], []]`
- `[[1]]`
- `[[32], [32], []]`
- `[[32], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path mistralai/Mistral-Small-4-119B-2603 --tp 1 --reasoning-parser mistral --tool-call-parser mistral
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
