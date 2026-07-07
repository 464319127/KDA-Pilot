# Profile evidence — minimax_m2__sglang_unified_attention_with_output

**Standalone kernel target: 5.7% of total serving GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M2`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `MiniMaxAI/MiniMax-M2` (slug `minimax_m2`, tp=4)
- Python interface: `sglang.unified_attention_with_output`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvFp16OFp16H128PagedKvCausalP64MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForG`, `fmhaSm100fKernel_QkvFp16OFp16H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvFp16OFp16H128PagedKvCausalP64VarSeqQ8Kv128PersistentSwapsAbForGen`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 4.53% |
| random | conc 100 | 4.71% |
| sharegpt | conc 1 | 4.58% |
| sharegpt | conc 32 | 2.83% |
| sharegpt | conc 100 | 5.66% |

**Peak: 5.7% in `sharegpt_high` (sharegpt, concurrency 100).**

## Input shapes (profiler)
- `[[0], [], [], []]`
- `[[100], [100], []]`
- `[[100]]`
- `[[104], [104], []]`
- `[[1622848, 2, 128], []]`
- `[[2560, 1536], [2560, 2, 128], [2560, 2, 128], [2560, 1536], [], [], [], [], [],`
- `[[2816, 1536], [2816, 2, 128], [2816, 2, 128], [2816, 1536], [], [], [], [], [],`
- `[[371], [], [], []]`
- `[[5632, 1536], [5632, 2, 128], [5632, 2, 128], [5632, 1536], [], [], [], [], [],`
- `[[640], [], [], []]`
- `[[699], [], [], []]`
- `[[[1]], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path MiniMaxAI/MiniMax-M2 --tp 4 --reasoning-parser minimax-append-think --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
