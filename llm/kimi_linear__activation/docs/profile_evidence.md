# Profile evidence — kimi_linear__activation

**Standalone kernel target: 3.4% of total serving GPU time** (max across scenarios) on
`moonshotai/Kimi-Linear-48B-A3B-Instruct`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

> Activation (SiLU/GELU+mul). Prior guidance: limited headroom — deprioritize.

- Model: `moonshotai/Kimi-Linear-48B-A3B-Instruct` (slug `kimi_linear`, tp=4)
- Python interface: `<confirm via capture; profiler family=activation>`
- Kernel family: `activation`  ·  Category: `other`
- GPU kernel(s): `void (anonymous namespace)::act_and_mul_kernel<__nv_bfloat16, ((anonymous namespace)::Acti`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 3.37% |

**Peak: 3.4% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[1, 38, 8, 128], [1, 38, 8, 128], []]`
- `[[48, 3072], [1, 48, 8, 128], [1, 48, 8], [1, 48, 8, 128], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct --tp 4 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
