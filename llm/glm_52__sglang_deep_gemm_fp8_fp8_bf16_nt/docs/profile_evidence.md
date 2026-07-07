# Profile evidence — glm_52__sglang_deep_gemm_fp8_fp8_bf16_nt

**Standalone kernel target: 28.6% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-5.2-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `zai-org/GLM-5.2-FP8` (slug `glm_52`, tp=8)
- Python interface: `sglang.deep_gemm_fp8_fp8_bf16_nt`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x16_64x16_2x4_2cta_h_bz_splitK_TNT`, `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 25.34% |
| random | conc 32 | 4.71% |
| random | conc 100 | 2.41% |
| sharegpt | conc 1 | 28.56% |
| sharegpt | conc 32 | 6.64% |
| sharegpt | conc 100 | 2.61% |

**Peak: 28.6% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[16, 1, 8, 512], [16, 1, 8, 512], []]`
- `[[16, 8, 512], [16, 1, 512], [16, 1, 512], [16, 4096], [], [], [16, 8, 64], [16,`
- `[[1], [1], []]`
- `[[1], []]`
- `[[2049, 1048580], [], []]`
- `[[32, 56], [32, 56], []]`
- `[[32], [32], []]`
- `[[48, 128], [], [], []]`
- `[[48, 4096], [], [], [], []]`
- `[[48, 8, 512], [48, 1, 512], [48, 1, 512], [48, 4096], [], [], [48, 8, 64], [48,`
- `[[704], [704], []]`
- `[[720], [], [], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path zai-org/GLM-5.2-FP8 --tp 8 --trust-remote-code --mem-fraction-static 0.8
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
