# Profile evidence — deepseek_v31__linear_gemm

**Standalone kernel target: 19.2% of total serving GPU time** (max across scenarios) on
`deepseek-ai/DeepSeek-V3.1`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `deepseek-ai/DeepSeek-V3.1` (slug `deepseek_v31`, tp=8)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_64x8_64x16_1x4_h_bz_splitK_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_splitK_TNT`, `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 19.19% |
| random | conc 32 | 4.89% |
| random | conc 100 | 7.25% |
| sharegpt | conc 1 | 5.50% |
| sharegpt | conc 100 | 11.95% |

**Peak: 19.2% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[1], [1], []]`
- `[[1], [], [], [], [], [], [], []]`
- `[[1]]`
- `[[28, 7168], [7168, 256]]`
- `[[39, 56]]`
- `[[39, 7168], [7168, 256]]`
- `[[49, 7168], [7168, 256]]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path deepseek-ai/DeepSeek-V3.1 --tp 8 --speculative-algorithm EAGLE --trust-remote-code
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
