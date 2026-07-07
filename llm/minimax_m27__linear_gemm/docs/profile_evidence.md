# Profile evidence — minimax_m27__linear_gemm

**Standalone kernel target: 12.5% of total serving GPU time** (max across scenarios) on
`MiniMaxAI/MiniMax-M2.7`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Profiler kernel-family; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `MiniMaxAI/MiniMax-M2.7` (slug `minimax_m27`, tp=8)
- Python interface: `<confirm via capture; profiler family=linear_gemm>`
- Kernel family: `linear_gemm`  ·  Category: `gemm`
- GPU kernel(s): `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x128x16_1x1x1_3_tnn_align1_bias_f32_relu`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_tnn_align1_bias_f32_relu`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_32x32x16_1x1x1_3_tnn_align1_bias_f32_relu`, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_tnn_align1_bias_f32_relu`, `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x`, `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 9.98% |
| random | conc 32 | 12.48% |
| random | conc 100 | 11.13% |
| sharegpt | conc 1 | 10.45% |
| sharegpt | conc 32 | 12.45% |
| sharegpt | conc 100 | 11.27% |

**Peak: 12.5% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[131], [], [], []]`
- `[[1536, 768], [1536, 1, 128], [1536, 1, 128], [1536, 768], [], [], [], [], [], [`
- `[[1], [], [], [], []]`
- `[[1]]`
- `[[204804], [], [], [], []]`
- `[[204804], [], [], []]`
- `[[212, 128], [212, 128], [4148736, 128], [4148736, 128], [212], [], [], []]`
- `[[330], [], [], []]`
- `[[388], [], [], [], []]`
- `[[408], [], [], []]`
- `[[467], [], [], []]`
- `[[54182], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path MiniMaxAI/MiniMax-M2.7 --tp 8 --ep 8 --tool-call-parser minimax-m2 --reasoning-parser minimax-append-think
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
