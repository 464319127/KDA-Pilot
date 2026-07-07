# Profile evidence — glm_45__attention

**Standalone kernel target: 9.1% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-4.5-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `zai-org/GLM-4.5-FP8` (slug `glm_45`, tp=8)
- Python interface: `<confirm via capture; profiler family=attention>`
- Kernel family: `attention`  ·  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_object_at__t`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.37% |
| random | conc 32 | 3.53% |
| random | conc 100 | 4.20% |
| sharegpt | conc 1 | 9.12% |
| sharegpt | conc 32 | 7.53% |
| sharegpt | conc 100 | 7.13% |

**Peak: 9.1% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[103, 12, 128], [19181, 128, 1, 128], [19181, 128, 1, 128], [], [2], [], [], [1`
- `[[1969, 12, 128], [19181, 128, 1, 128], [19181, 128, 1, 128], [], [39], [], [], `
- `[[1], [], [], [], [], [], []]`
- `[[1], []]`
- `[[1]]`
- `[[2276, 12, 128], [19181, 128, 1, 128], [19181, 128, 1, 128], [], [48], [], [], `
- `[[2444, 12, 128], [19181, 128, 1, 128], [19181, 128, 1, 128], [], [43], [], [], `
- `[[2455168, 1, 128], []]`
- `[[7680, 1, 128], [], [], []]`
- `[[80, 12, 128], [19181, 128, 1, 128], [19181, 128, 1, 128], [], [2], [], [], [1]`
- `[[8125, 12, 128], [19181, 128, 1, 128], [19181, 128, 1, 128], [], [27], [], [], `
- `[[8125, 1536], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path zai-org/GLM-4.5-FP8 --tp 8 --reasoning-parser glm45 --tool-call-parser glm45 --attention-backend fa4
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
