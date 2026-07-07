# Profile evidence — glm_5__sglang_flashinfer_fp4_quantize

**Standalone kernel target: 13.1% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-5-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `nvidia/GLM-5-NVFP4` (slug `glm_5`, tp=4)
- Python interface: `sglang.flashinfer_fp4_quantize`
- Kernel family: `attention`  ·  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16H256SeparateQkvCausalVarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1Mult`, `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1VarSeqQ16Kv128P`, `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPer`, `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeSwizzledKern`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 11.86% |
| random | conc 32 | 2.03% |
| random | conc 100 | 4.89% |
| sharegpt | conc 1 | 11.40% |
| sharegpt | conc 32 | 8.39% |
| sharegpt | conc 100 | 13.10% |

**Peak: 13.1% in `sharegpt_high` (sharegpt, concurrency 100).**

## Input shapes (profiler)
- `[[1, 1], [1, 1], []]`
- `[[16873, 16, 256], []]`
- `[[16873, 3072], [3072, 6144], [16896, 384], [384, 6144], [], [], []]`
- `[[1], [1], []]`
- `[[1], [], [], []]`
- `[[1], [], []]`
- `[[201]]`
- `[[20784, 1, 16, 512], []]`
- `[[20784, 1, 16, 576], []]`
- `[[30796, 3072], [3072, 1024], [30848, 384], [384, 1024], [], [], []]`
- `[[30796, 3072], [3072, 6144], [30848, 384], [384, 6144], [], [], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path nvidia/GLM-5-NVFP4 --tp 4 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
