# Profile evidence — kimi_k2__sglang_unified_attention_with_output

**Standalone kernel target: 9.5% of total serving GPU time** (max across scenarios) on
`moonshotai/Kimi-K2-Instruct`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `moonshotai/Kimi-K2-Instruct` (slug `kimi_k2`, tp=8)
- Python interface: `sglang.unified_attention_with_output`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512HVPerCta256PagedKvDenseP64MultiCtasKvVarSe`, `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512PagedKvDenseP64MultiCtasKvVarSeqQ8Kv128Sta`, `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk576HV512PagedKvDenseP64VarSeqQ8Kv128PersistentSwap`, `void flashinfer::mla::BatchMLAPagedAttentionKernel<flashinfer::mla::KernelTraits<true, 2u,`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.73% |
| random | conc 100 | 2.62% |
| sharegpt | conc 1 | 6.01% |
| sharegpt | conc 32 | 2.40% |
| sharegpt | conc 100 | 9.49% |

**Peak: 9.5% in `sharegpt_high` (sharegpt, concurrency 100).**

## Input shapes (profiler)
- `[[1, 163840], [], []]`
- `[[1536, 8, 512], [1536, 1, 512], [1536, 1, 512], [1536, 4096], [], [], [1536, 8,`
- `[[1]]`
- `[[2112], [2112], []]`
- `[[255], [], [], []]`
- `[[25], [], [], []]`
- `[[320], [320], []]`
- `[[50], [], [], []]`
- `[[512], [], [], []]`
- `[[543], [], [], []]`
- `[[[1]], [], [], [], [], []]`
- `[[[320], [38]], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path moonshotai/Kimi-K2-Instruct --tp 8 --tool-call-parser kimi_k2
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
