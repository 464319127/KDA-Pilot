# Profile evidence — gemma4__attention

**Standalone kernel target: 12.0% of total serving GPU time** (max across scenarios) on
`google/gemma-4-26B-A4B-it`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `google/gemma-4-26B-A4B-it` (slug `gemma4`, tp=1)
- Python interface: `<confirm via capture; profiler family=attention>`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvSlidingOrChunkedCausalP64MultiCtasKvCgaVar`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvSlidingOrChunkedCausalP64VarSeqQ128Kv128Pe`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvSlidingOrChunkedCausalP64VarSeqQ8Kv128Pers`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 4.07% |
| random | conc 32 | 5.69% |
| random | conc 100 | 12.04% |
| sharegpt | conc 1 | 4.14% |
| sharegpt | conc 32 | 6.32% |
| sharegpt | conc 100 | 7.92% |

**Peak: 12.0% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[1, 262144], [], [], []]`
- `[[100]]`
- `[[104, 262144], [], [], []]`
- `[[1]]`
- `[[2816, 4096]]`
- `[[32, 16], [32, 16], []]`
- `[[32, 262144], [], [], []]`
- `[[32]]`
- `[[384]]`
- `[[51], [], [], []]`
- `[[[1], [1], [1], [1]], [[1], [1], [1], [1]], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path google/gemma-4-26B-A4B-it --reasoning-parser gemma4 --tool-call-parser gemma4
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
