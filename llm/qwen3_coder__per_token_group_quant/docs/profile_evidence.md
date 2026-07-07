# Profile evidence — qwen3_coder__per_token_group_quant

**Standalone kernel target: 17.2% of total serving GPU time** (max across scenarios) on
`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` (slug `qwen3_coder`, tp=8)
- Python interface: `<confirm via capture; profiler family=per_token_group_quant>`
- Kernel family: `per_token_group_quant`  ·  Category: `quant_gemm`
- GPU kernel(s): `void per_token_group_quant_8bit_kernel<NaiveScheduler, 128, 8, __nv_bfloat16, c10::Float8_`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 17.25% |
| random | conc 100 | 8.00% |
| sharegpt | conc 1 | 16.25% |
| sharegpt | conc 32 | 7.58% |

**Peak: 17.2% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[17, 1536], [17, 1536], []]`
- `[[20, 1536], [20, 1, 128], [20, 1, 128], [20, 1536], [], [], [], [], [], [], [],`
- `[[2825120, 1, 128], []]`
- `[[31], [31], []]`
- `[[6, 1536], [6, 1536], []]`
- `[[8, 1536], [8, 1, 128], [8, 1, 128], [8, 1536], [], [], [], [], [], [], [], [],`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --tp 8 --ep 8 --tool-call-parser qwen3_coder
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
