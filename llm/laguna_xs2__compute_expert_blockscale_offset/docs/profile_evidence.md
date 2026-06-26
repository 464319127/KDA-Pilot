# Profile evidence — laguna_xs2__compute_expert_blockscale_offset

**e2e-optimization target: 7.1% of total GPU time** (max across scenarios) on
`poolside/Laguna-XS.2-NVFP4`, from the exact cookbook-aligned profile. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

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

## Reproduce (cookbook-aligned)
```bash
python -m sglang.launch_server --model-path poolside/Laguna-XS.2-NVFP4 --tp 1 --trust-remote-code
```
After optimizing, re-run **random_low** to validate the e2e effect.
