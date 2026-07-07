# Profile evidence — glm_52__per_token_group_quant

**Standalone kernel target: 10.9% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-5.2-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `zai-org/GLM-5.2-FP8` (slug `glm_52`, tp=8)
- Python interface: `<confirm via capture; profiler family=per_token_group_quant>`
- Kernel family: `per_token_group_quant`  ·  Category: `quant_gemm`
- GPU kernel(s): `void (anonymous namespace)::per_token_group_quant_8bit_v2_kernel<(anonymous namespace)::Na`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 10.92% |
| random | conc 100 | 5.25% |
| sharegpt | conc 1 | 10.74% |
| sharegpt | conc 32 | 5.52% |
| sharegpt | conc 100 | 8.78% |

**Peak: 10.9% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[16, 1, 8, 512], [16, 1, 8, 512], []]`
- `[[16, 8, 512], [16, 1, 512], [16, 1, 512], [16, 4096], [], [], [16, 8, 64], [16,`
- `[[48, 8, 512], [48, 1, 512], [48, 1, 512], [48, 4096], [], [], [48, 8, 64], [48,`
- `[[], [144, 128], [144, 32, 128], [144, 32, 1], [144, 2048]]`
- `[[], [16, 128], [16, 32, 128], [16, 32, 1], [16, 2048]]`
- `[[], [48, 128], [48, 32, 128], [48, 32, 1], [48, 2048]]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path zai-org/GLM-5.2-FP8 --tp 8 --trust-remote-code --mem-fraction-static 0.8
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
