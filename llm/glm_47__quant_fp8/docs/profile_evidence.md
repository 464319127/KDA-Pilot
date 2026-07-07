# Profile evidence — glm_47__quant_fp8

**Standalone kernel target: 14.2% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-4.7-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/GLM-4.7-NVFP4` (slug `glm_47`, tp=8)
- Python interface: `<confirm via capture; profiler family=quant_fp8>`
- Kernel family: `quant_fp8`  ·  Category: `quant_gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeSwizzledKern`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 14.17% |
| random | conc 32 | 5.02% |
| random | conc 100 | 6.41% |
| sharegpt | conc 1 | 13.94% |
| sharegpt | conc 32 | 5.54% |
| sharegpt | conc 100 | 8.08% |

**Peak: 14.2% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[16, 1536], [16, 1, 128], [16, 1, 128], [16, 1536], [], [], [], [], [], [], [],`
- `[[16, 1536], [16, 1536], []]`
- `[[16, 1536], [], [], []]`
- `[[39, 1536], [39, 1536], []]`
- `[[48, 1536], [48, 1, 128], [48, 1, 128], [48, 1536], [], [], [], [], [], [], [],`
- `[[5299776, 1, 128], []]`
- `[[82809, 1, 64, 128], [], [], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model nvidia/GLM-4.7-NVFP4 --tp 8 --quantization modelopt_fp4 --reasoning-parser glm45 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
