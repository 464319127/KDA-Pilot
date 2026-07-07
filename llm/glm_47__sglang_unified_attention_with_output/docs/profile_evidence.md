# Profile evidence — glm_47__sglang_unified_attention_with_output

**Standalone kernel target: 4.9% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-4.7-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `nvidia/GLM-4.7-NVFP4` (slug `glm_47`, tp=8)
- Python interface: `sglang.unified_attention_with_output`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvE4m3OBfloat16H128PagedKvCausalP64MultiCtasKvVarSeqQ16Kv128StaticSwapsA`, `fmhaSm100fKernel_QkvE4m3OBfloat16H128PagedKvCausalP64VarSeqQ128Kv128PersistentContext`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 4.64% |
| random | conc 100 | 3.69% |
| sharegpt | conc 1 | 4.87% |
| sharegpt | conc 32 | 3.32% |
| sharegpt | conc 100 | 4.34% |

**Peak: 4.9% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[1024, 1536], [1024, 1, 128], [1024, 1, 128], [1024, 1536], [], [], [], [], [],`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1], []]`
- `[[5299776, 1, 128], []]`
- `[[64], [64], []]`
- `[[704], [], [], [], []]`
- `[[82809, 1, 64, 128], [], [], []]`
- `[[82809, 64, 1, 128], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model nvidia/GLM-4.7-NVFP4 --tp 8 --quantization modelopt_fp4 --reasoning-parser glm45 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
