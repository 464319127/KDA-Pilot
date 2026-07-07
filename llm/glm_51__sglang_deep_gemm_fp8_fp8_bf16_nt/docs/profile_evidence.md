# Profile evidence — glm_51__sglang_deep_gemm_fp8_fp8_bf16_nt

**Standalone kernel target: 20.6% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-5.1-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `zai-org/GLM-5.1-FP8` (slug `glm_51`, tp=8)
- Python interface: `sglang.deep_gemm_fp8_fp8_bf16_nt`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_splitK_TNT`, `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 20.55% |
| random | conc 32 | 8.43% |
| random | conc 100 | 9.96% |
| sharegpt | conc 1 | 18.18% |
| sharegpt | conc 32 | 13.60% |
| sharegpt | conc 100 | 12.02% |

**Peak: 20.6% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[16, 6144], [6144, 256]]`
- `[[38, 6144], [6144, 256]]`
- `[[4, 12288], [12288, 6144]]`
- `[[448], [], [], []]`
- `[[49, 1], [], [], []]`
- `[[49], [49], []]`
- `[[6, 6144], [6144, 256]]`
- `[[[0], [38]], []]`
- `[[]]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path zai-org/GLM-5.1-FP8 --tp 8 --tool-call-parser glm47 --reasoning-parser glm45
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
