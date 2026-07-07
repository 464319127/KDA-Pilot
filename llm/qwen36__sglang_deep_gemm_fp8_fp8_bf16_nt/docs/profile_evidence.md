# Profile evidence — qwen36__sglang_deep_gemm_fp8_fp8_bf16_nt

**Standalone kernel target: 26.3% of total serving GPU time** (max across scenarios) on
`Qwen/Qwen3.6-35B-A3B-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

- Model: `Qwen/Qwen3.6-35B-A3B-FP8` (slug `qwen36`, tp=1)
- Python interface: `sglang.deep_gemm_fp8_fp8_bf16_nt`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_192x8_64x8_2x1_v_bz_TNT`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 25.87% |
| random | conc 32 | 5.76% |
| random | conc 100 | 4.87% |
| sharegpt | conc 1 | 26.34% |
| sharegpt | conc 32 | 6.94% |
| sharegpt | conc 100 | 2.12% |

**Peak: 26.3% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[1, 1], [1, 1], []]`
- `[[1, 1], [], [], [], [], []]`
- `[[1, 4], []]`
- `[[11886, 2048], [11886, 4], [12288, 2048], [12288, 4], [11886, 12288]]`
- `[[15434, 2048], [15434, 4], [12288, 2048], [12288, 4], [15434, 12288]]`
- `[[17, 2048], [17, 4], [9216, 2048], [9216, 4], [17, 9216]]`
- `[[192], [], [], []]`
- `[[32, 16], [32, 16], []]`
- `[[384], [384], []]`
- `[[44, 248320], [], []]`
- `[[49, 2], []]`
- `[[49], [49], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path Qwen/Qwen3.6-35B-A3B-FP8 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
