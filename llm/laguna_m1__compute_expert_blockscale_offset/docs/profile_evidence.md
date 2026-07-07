# Profile evidence — laguna_m1__compute_expert_blockscale_offset

**Standalone kernel target: 5.7% of total serving GPU time** (max across scenarios) on
`poolside/Laguna-M.1-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `poolside/Laguna-M.1-NVFP4` (slug `laguna_m1`, tp=8)
- Python interface: `<confirm via capture; profiler family=compute_expert_blockscale_offset>`
- Kernel family: `compute_expert_blockscale_offset`  ·  Category: `moe`
- GPU kernel(s): `compute_expert_blockscale_offsets(int const*, int*, int*, int*, long)`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.73% |
| random | conc 100 | 2.22% |
| sharegpt | conc 1 | 5.64% |
| sharegpt | conc 100 | 2.33% |

**Peak: 5.7% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[256], [256], []]`
- `[[2816, 1, 128], [], [], []]`
- `[[4097, 262148], []]`
- `[[768], [768], []]`
- `[[99], [], []]`
- `[[[256]], []]`
- `[[], [], [], [], [], []]`
- `[[]]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path poolside/Laguna-M.1-NVFP4 --tp 8 --trust-remote-code --reasoning-parser poolside_v1 --tool-call-parser poolside_v1
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
