# Profile evidence — minimax_m25__sglang_unified_attention_with_output

**Standalone kernel target: 5.2% of total serving GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M2.5`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `MiniMaxAI/MiniMax-M2.5` (slug `minimax_m25`, tp=8)
- Python interface: `sglang.unified_attention_with_output`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvFp16OFp16H128PagedKvCausalP64MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForG`, `fmhaSm100fKernel_QkvFp16OFp16H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 3.98% |
| random | conc 100 | 3.61% |
| sharegpt | conc 1 | 4.09% |
| sharegpt | conc 32 | 2.70% |
| sharegpt | conc 100 | 5.20% |

**Peak: 5.2% in `sharegpt_high` (sharegpt, concurrency 100).**

## Input shapes (profiler)
- `[[196612], [], [], []]`
- `[[1]]`
- `[[256], [], [], [], []]`
- `[[4097], [], [], [], []]`
- `[[512, 768], [], [], [], []]`
- `[[5524, 1, 128], []]`
- `[[768, 768], [768, 1, 128], [768, 1, 128], [768, 768], [], [], [], [], [], [], [`
- `[[[1]], [], [], [], [], []]`
- `[[[640], [59]], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path MiniMaxAI/MiniMax-M2.5 --tp 8 --ep 8 --reasoning-parser minimax-append-think
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
