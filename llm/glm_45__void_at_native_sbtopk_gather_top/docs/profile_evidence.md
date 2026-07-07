# Profile evidence — glm_45__void_at_native_sbtopk_gather_top

**Standalone kernel target: 5.9% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-4.5-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `zai-org/GLM-4.5-FP8` (slug `glm_45`, tp=8)
- Python interface: `<confirm via capture; profiler family=void_at_native_sbtopk_gather_top>`
- Kernel family: `void_at_native_sbtopk_gather_top`  ·  Category: `moe`
- GPU kernel(s): `void at::native::sbtopk::gatherTopK<float, unsigned int, 2, false>(at::cuda::detail::Tenso`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 5.92% |
| random | conc 100 | 5.84% |
| sharegpt | conc 32 | 5.75% |
| sharegpt | conc 100 | 5.05% |

**Peak: 5.9% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[1287, 12, 128]]`
- `[[1], []]`
- `[[2276, 1, 128], []]`
- `[[2276, 12, 128], [19181, 128, 1, 128], [19181, 128, 1, 128], [], [48], [], [], `
- `[[2444, 12, 128]]`
- `[[2560, 1536], [2560, 1, 128], [2560, 1, 128], [2560, 1536], [], [], [], [], [],`
- `[[7639, 1536], []]`
- `[[8192, 1536], [8192, 1, 128], [8192, 1, 128], [8192, 1536], [], [], [], [], [],`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path zai-org/GLM-4.5-FP8 --tp 8 --reasoning-parser glm45 --tool-call-parser glm45 --attention-backend fa4
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
