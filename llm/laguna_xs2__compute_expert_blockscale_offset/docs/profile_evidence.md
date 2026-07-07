# Profile evidence — laguna_xs2__compute_expert_blockscale_offset

**Standalone kernel target: 7.1% of total serving GPU time** (max across scenarios) on
`poolside/Laguna-XS.2-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `poolside/Laguna-XS.2-NVFP4` (slug `laguna_xs2`, tp=1)
- Python interface: `<confirm via capture; profiler family=compute_expert_blockscale_offset>`
- Kernel family: `compute_expert_blockscale_offset`  ·  Category: `moe`
- GPU kernel(s): `compute_expert_blockscale_offsets(int const*, int*, int*, int*, long)`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 7.13% |
| random | conc 32 | 2.47% |
| random | conc 100 | 3.32% |
| sharegpt | conc 1 | 7.10% |
| sharegpt | conc 100 | 4.33% |

**Peak: 7.1% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[14], [], [], []]`
- `[[1], [1], []]`
- `[[1], [], []]`
- `[[1], []]`
- `[[257], [], [], []]`
- `[[29], [29], []]`
- `[[32], [32], []]`
- `[[48, 8192], [48, 8, 128], [48, 8, 128], [48, 8192], [], [], [], [], [], [], [],`
- `[[664], [], [], []]`
- `[[832], [], [], []]`
- `[[[1088]], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path poolside/Laguna-XS.2-NVFP4 --tp 1 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
