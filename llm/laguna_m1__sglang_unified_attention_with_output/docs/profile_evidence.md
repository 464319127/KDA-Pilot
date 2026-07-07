# Profile evidence — laguna_m1__sglang_unified_attention_with_output

**Standalone kernel target: 8.0% of total serving GPU time** (max across scenarios) on
`poolside/Laguna-M.1-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `poolside/Laguna-M.1-NVFP4` (slug `laguna_m1`, tp=8)
- Python interface: `sglang.unified_attention_with_output`
- Kernel family: `attention`  ·  Category: `gemm`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64MultiCtasKvVarSeqQ8Kv128StaticSwa`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP64VarSeqQ8Kv128PersistentSwapsAbFor`, `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPer`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 7.96% |
| sharegpt | conc 1 | 7.95% |
| sharegpt | conc 100 | 7.13% |

**Peak: 8.0% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[128], [], [], []]`
- `[[1], [1], []]`
- `[[1]]`
- `[[2276, 1024], [2276, 1024], []]`
- `[[2304, 1024], [2304, 1, 128], [2304, 1, 128], [2304, 1024], [], [], [], [], [],`
- `[[64], [], [], [], [], [], []]`
- `[[768], [768], []]`
- `[[[1]], [], [], [], [], []]`
- `[[]]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path poolside/Laguna-M.1-NVFP4 --tp 8 --trust-remote-code --reasoning-parser poolside_v1 --tool-call-parser poolside_v1
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
