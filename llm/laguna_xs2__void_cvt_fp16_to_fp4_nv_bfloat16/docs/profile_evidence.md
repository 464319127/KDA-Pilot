# Profile evidence — laguna_xs2__void_cvt_fp16_to_fp4_nv_bfloat16

**Standalone kernel target: 9.2% of total serving GPU time** (max across scenarios) on
`poolside/Laguna-XS.2-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `poolside/Laguna-XS.2-NVFP4` (slug `laguna_xs2`, tp=1)
- Python interface: `<confirm via capture; profiler family=void_cvt_fp16_to_fp4_nv_bfloat16>`
- Kernel family: `void_cvt_fp16_to_fp4_nv_bfloat16`  ·  Category: `quant_gemm`
- GPU kernel(s): `void cvt_fp16_to_fp4<__nv_bfloat16, false, false>(int, int, __nv_bfloat16 const*, float co`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 9.17% |
| random | conc 100 | 6.27% |
| sharegpt | conc 100 | 6.06% |

**Peak: 9.2% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[0], [0], []]`
- `[[12663, 64, 8, 128], [], [], []]`
- `[[12663, 64, 8, 128], []]`
- `[[13, 100352], [], []]`
- `[[144, 8192], [144, 8, 128], [144, 8, 128], [144, 8192], [], [], [], [], [], [],`
- `[[1], [1], []]`
- `[[316], [], [], []]`
- `[[32], [32], []]`
- `[[534, 8192], [534, 8192], []]`
- `[[64], [], [], []]`
- `[[[1280], [64]], []]`
- `[[[384], [2]], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path poolside/Laguna-XS.2-NVFP4 --tp 1 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
