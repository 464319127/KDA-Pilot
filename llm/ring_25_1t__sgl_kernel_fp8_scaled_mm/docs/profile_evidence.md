# Profile evidence — ring_25_1t__sgl_kernel_fp8_scaled_mm

**Standalone kernel target: 17.7% of total serving GPU time** (max across scenarios) on
`inclusionAI/Ring-2.5-1T`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `inclusionAI/Ring-2.5-1T` (slug `ring_25_1t`, tp=8)
- Python interface: `sgl_kernel.fp8_scaled_mm`
- Kernel family: `linear_gemm`  ·  Category: `gemm`
- GPU kernel(s): `_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10colle`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x128x16_1x1x1_3_tnn_align1_bias_f32_relu`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_tnn_align1_bias_f32_relu`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 7.40% |
| random | conc 32 | 13.92% |
| random | conc 100 | 17.69% |
| sharegpt | conc 1 | 7.52% |
| sharegpt | conc 32 | 15.97% |
| sharegpt | conc 100 | 7.47% |

**Peak: 17.7% in `random_high` (random, concurrency 100).**

## Input shapes (profiler)
- `[[14375, 8192], [8192, 256]]`
- `[[14375, 8192], [8192, 4608], [14375, 1], [4608, 1], [], []]`
- `[[16250, 8192], [8192, 4608], [16250, 1], [4608, 1], [], []]`
- `[[16384, 8192], [8192, 256]]`
- `[[16384, 8192], [8192, 4608], [16384, 1], [4608, 1], [], []]`
- `[[189, 8192], [8192, 4608], [189, 1], [4608, 1], [], []]`
- `[[1], [], [], [], [], [], []]`
- `[[1]]`
- `[[32], [32], []]`
- `[[32]]`
- `[[3330, 8192], [8192, 256]]`
- `[[3330, 8192], [8192, 4608], [3330, 1], [4608, 1], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path inclusionAI/Ring-2.5-1T --tp 8 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
