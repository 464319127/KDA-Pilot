# Profile evidence — deepseek_math_v2__void_anonymous_namespace_fast_ha

**Standalone kernel target: 3.8% of total serving GPU time** (max across scenarios) on
`deepseek-ai/DeepSeek-Math-V2`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `deepseek-ai/DeepSeek-Math-V2` (slug `deepseek_math_v2`, tp=8)
- Python interface: `<confirm via capture; profiler family=void_anonymous_namespace_fast_ha>`
- Kernel family: `void_anonymous_namespace_fast_ha`  ·  Category: `other`
- GPU kernel(s): `void (anonymous namespace)::fast_hadamard_transform_kernel<(anonymous namespace)::FastHada`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 100 | 2.23% |
| sharegpt | conc 32 | 3.80% |
| sharegpt | conc 100 | 2.17% |

**Peak: 3.8% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[8661, 64, 128], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path deepseek-ai/DeepSeek-Math-V2 --tp 8 --ep 8 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
