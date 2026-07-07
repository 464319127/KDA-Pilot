# Profile evidence — deepseek_v4__mhc_post_tilelang_kernel

**Standalone kernel target: 4.2% of total serving GPU time** (max across scenarios) on
`deepseek-ai/DeepSeek-V4-Flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `deepseek-ai/DeepSeek-V4-Flash` (slug `deepseek_v4`, tp=4)
- Python interface: `<confirm via capture; profiler family=mhc_post_tilelang_kernel>`
- Kernel family: `mhc_post_tilelang_kernel`  ·  Category: `other`
- GPU kernel(s): `mhc_post_tilelang_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 4.23% |

**Peak: 4.2% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[38, 1, 64, 512], [1988, 256, 1, 584], [38, 1, 128], [38], [64], [148, 8], [39]`
- `[[38, 4, 4096], []]`
- `[[38, 4096], [38, 8], [1536, 4096], [1536, 8], [38, 1536]]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path deepseek-ai/DeepSeek-V4-Flash --tp 4 --moe-runner-backend flashinfer_mxfp4 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
