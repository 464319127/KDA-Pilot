# Profile evidence — ring_25_1t__void_moe_sum_reduce_warp_per_tok

**Standalone kernel target: 3.3% of total serving GPU time** (max across scenarios) on
`inclusionAI/Ring-2.5-1T`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `inclusionAI/Ring-2.5-1T` (slug `ring_25_1t`, tp=8)
- Python interface: `<confirm via capture; profiler family=void_moe_sum_reduce_warp_per_tok>`
- Kernel family: `void_moe_sum_reduce_warp_per_tok`  ·  Category: `moe`
- GPU kernel(s): `void moe_sum_reduce_warp_per_token_vec_kernel<8>(c10::BFloat16 const*, c10::BFloat16*, lon`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 100 | 3.25% |
| sharegpt | conc 32 | 2.26% |

**Peak: 3.3% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[14375, 8, 8192], [14375, 8192], []]`
- `[[16384, 8, 8192], [16384, 8192], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path inclusionAI/Ring-2.5-1T --tp 8 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
