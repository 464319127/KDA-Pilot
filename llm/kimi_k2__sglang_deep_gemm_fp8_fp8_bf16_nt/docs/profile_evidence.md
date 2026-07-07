# Profile evidence — kimi_k2__sglang_deep_gemm_fp8_fp8_bf16_nt

**Standalone kernel target: 23.1% of total serving GPU time** (max across scenarios) on
`moonshotai/Kimi-K2-Instruct`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `moonshotai/Kimi-K2-Instruct` (slug `kimi_k2`, tp=8)
- Python interface: `sglang.deep_gemm_fp8_fp8_bf16_nt`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_8x64_64x16_4x1_v_bz_TNN`, `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 18.82% |
| random | conc 32 | 13.60% |
| sharegpt | conc 1 | 23.12% |
| sharegpt | conc 32 | 3.20% |
| sharegpt | conc 100 | 2.23% |

**Peak: 23.1% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[1], [1], []]`
- `[[1], [], [], []]`
- `[[1]]`
- `[[256], [], [], [], []]`
- `[[263], [], [], []]`
- `[[280], [], [], [], []]`
- `[[32, 163840], [], []]`
- `[[48, 8, 512], [48, 1, 512], [48, 1, 512], [48, 4096], [], [], [48, 8, 64], [48,`
- `[[48], [], [], [], []]`
- `[[6], [], [], []]`
- `[[8084, 2304], [8084, 5], [7168, 2304], [7168, 5], [8084, 7168]]`
- `[[8084, 7168], [8084, 14], [4608, 7168], [4608, 14], [8084, 4608]]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path moonshotai/Kimi-K2-Instruct --tp 8 --tool-call-parser kimi_k2
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
