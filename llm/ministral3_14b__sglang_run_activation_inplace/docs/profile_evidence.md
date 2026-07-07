# Profile evidence — ministral3_14b__sglang_run_activation_inplace

**Standalone kernel target: 5.3% of total serving GPU time** (max across scenarios) on
`mistralai/Ministral-3-14B-Instruct-2512`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

> Activation (SiLU/GELU+mul). Prior guidance: limited headroom — deprioritize.

- Model: `mistralai/Ministral-3-14B-Instruct-2512` (slug `ministral3_14b`, tp=1)
- Python interface: `sglang._run_activation_inplace`
- Kernel family: `activation`  ·  Category: `other`
- GPU kernel(s): `void (anonymous namespace)::act_and_mul_kernel<__nv_bfloat16, ((anonymous namespace)::Acti`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 2.84% |
| random | conc 32 | 5.33% |
| random | conc 100 | 4.13% |
| sharegpt | conc 32 | 5.06% |
| sharegpt | conc 100 | 4.13% |

**Peak: 5.3% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[1], [1], []]`
- `[[], [1228, 32768], [1228, 16384]]`
- `[[], [1316, 32768], [1316, 16384]]`
- `[[], [39, 32768], [39, 16384]]`
- `[[], [6642, 32768], [6642, 16384]]`
- `[[], [8384, 32768], [8384, 16384]]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path mistralai/Ministral-3-14B-Instruct-2512 --tp 1 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
