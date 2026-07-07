# Profile evidence — kimi_k27_code__linear_gemm

**Standalone kernel target: 38.4% of total serving GPU time** (max across scenarios) on
`moonshotai/Kimi-K2.7-Code`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `moonshotai/Kimi-K2.7-Code` (slug `kimi_k27_code`, tp=8)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x24_64x16_1x2_h_bz_splitK_TNT`, `nvjet_sm100_tst_64x32_64x16_2x1_2cta_v_bz_splitK_TNT`, `nvjet_sm100_tst_64x8_64x16_1x2_h_bz_splitK_TNT`, `nvjet_sm100_tst_64x8_64x16_2x2_h_bz_splitK_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`, `void fused_a_gemm_kernel<1, 2112, 7168, 16, 8, 256, 16>(__nv_bfloat16*, __nv_bfloat16 cons`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 38.36% |
| random | conc 32 | 11.24% |
| random | conc 100 | 23.34% |
| sharegpt | conc 1 | 38.40% |
| sharegpt | conc 32 | 20.12% |
| sharegpt | conc 100 | 26.55% |

**Peak: 38.4% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[1, 163840], [], []]`
- `[[1, 2], [], []]`
- `[[1, 31], [], []]`
- `[[1536, 1536], [], []]`
- `[[1], [], [], [], [], [], []]`
- `[[1], [], [], []]`
- `[[1], [], []]`
- `[[1]]`
- `[[2048, 512], [], []]`
- `[[2048, 512]]`
- `[[384, 1, 64], []]`
- `[[39, 384], [39, 384], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path moonshotai/Kimi-K2.7-Code --tp 8 --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
