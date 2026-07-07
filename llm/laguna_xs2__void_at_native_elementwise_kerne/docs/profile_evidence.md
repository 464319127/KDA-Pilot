# Profile evidence — laguna_xs2__void_at_native_elementwise_kerne

**Standalone kernel target: 4.6% of total serving GPU time** (max across scenarios) on
`poolside/Laguna-XS.2-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `poolside/Laguna-XS.2-NVFP4` (slug `laguna_xs2`, tp=1)
- Python interface: `<confirm via capture; profiler family=void_at_native_elementwise_kerne>`
- Kernel family: `void_at_native_elementwise_kerne`  ·  Category: `memory_bound`
- GPU kernel(s): `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native:`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 4.60% |
| random | conc 100 | 2.13% |

**Peak: 4.6% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[1], [1], []]`
- `[[4717, 8192], [4717, 8192], []]`
- `[[511], [], [], []]`
- `[[5120, 8192], [5120, 8, 128], [5120, 8, 128], [5120, 8192], [], [], [], [], [],`
- `[[704], [], [], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path poolside/Laguna-XS.2-NVFP4 --tp 1 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
