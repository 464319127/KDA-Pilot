# Profile evidence — glm_46__void_moe_sum_reduce_warp_per_tok

**Standalone kernel target: 3.4% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-4.6-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `zai-org/GLM-4.6-FP8` (slug `glm_46`, tp=8)
- Python interface: `<confirm via capture; profiler family=void_moe_sum_reduce_warp_per_tok>`
- Kernel family: `void_moe_sum_reduce_warp_per_tok`  ·  Category: `moe`
- GPU kernel(s): `void moe_sum_reduce_warp_per_token_vec_kernel<8>(c10::BFloat16 const*, c10::BFloat16*, lon`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 3.38% |
| random | conc 100 | 2.58% |
| sharegpt | conc 32 | 3.13% |
| sharegpt | conc 100 | 2.13% |

**Peak: 3.4% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[1], []]`
- `[[2193, 12, 128]]`
- `[[2816, 1, 128], [], [], []]`
- `[[8125, 12, 128], [19176, 128, 1, 128], [19176, 128, 1, 128], [], [27], [], [], `
- `[[[0], [55]], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path zai-org/GLM-4.6-FP8 --tp 8 --reasoning-parser glm45 --tool-call-parser glm45 --attention-backend fa4
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
