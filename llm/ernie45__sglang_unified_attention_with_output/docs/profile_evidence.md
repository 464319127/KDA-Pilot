# Profile evidence — ernie45__sglang_unified_attention_with_output

**Standalone kernel target: 8.0% of total serving GPU time** (max across scenarios) on
`baidu/ERNIE-4.5-21B-A3B-PT`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `baidu/ERNIE-4.5-21B-A3B-PT` (slug `ernie45`, tp=1)
- Python interface: `sglang.unified_attention_with_output`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64MultiCtasKvVarSeqQ8Kv128StaticSwa`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64VarSeqQ8Kv128PersistentSwapsAbFor`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.83% |
| random | conc 32 | 2.35% |
| random | conc 100 | 8.02% |
| sharegpt | conc 1 | 5.96% |
| sharegpt | conc 32 | 4.07% |
| sharegpt | conc 100 | 7.81% |

**Peak: 8.0% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[0], [], [], []]`
- `[[100]]`
- `[[104, 17], [104, 17], []]`
- `[[11264, 2560], [11264, 4, 128], [11264, 4, 128], [11264, 2560], [], [], [], [],`
- `[[1], [1], []]`
- `[[1]]`
- `[[256], [256], []]`
- `[[2816, 2560], [2816, 4, 128], [2816, 4, 128], [2816, 2560], [], [], [], [], [],`
- `[[5120, 2560], [5120, 4, 128], [5120, 4, 128], [5120, 2560], [], [], [], [], [],`
- `[[55], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path baidu/ERNIE-4.5-21B-A3B-PT --tp 1
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
