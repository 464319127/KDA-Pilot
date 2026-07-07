# Profile evidence — deepseek_v32__void_anonymous_namespace_fast_ha

**Standalone kernel target: 5.1% of total serving GPU time** (max across scenarios) on
`nvidia/DeepSeek-V3.2-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/DeepSeek-V3.2-NVFP4` (slug `deepseek_v32`, tp=4)
- Python interface: `<confirm via capture; profiler family=void_anonymous_namespace_fast_ha>`
- Kernel family: `void_anonymous_namespace_fast_ha`  ·  Category: `other`
- GPU kernel(s): `void (anonymous namespace)::fast_hadamard_transform_kernel<(anonymous namespace)::FastHada`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| sharegpt | conc 32 | 5.05% |
| sharegpt | conc 100 | 2.20% |

**Peak: 5.1% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[13312, 7168], [], [], [], []]`
- `[[2048, 32, 512], [2048, 1, 512], [2048, 1, 512], [2048, 16384], [], [], [2048, `
- `[[], [2048, 128], [2048, 64, 128], [2048, 64, 1], [2048, 2048]]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model nvidia/DeepSeek-V3.2-NVFP4 --tp 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
