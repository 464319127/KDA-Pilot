# Profile evidence — glm_52__sglang_deep_gemm_fp8_fp8_bf16_nt

**Standalone kernel target: 28.6% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-5.2-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below were recaptured from a real GLM-5.2-FP8 TP=8 server run and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-5.2-FP8` (slug `glm_52`, tp=8)
- Python interface(s): `sglang.srt.layers.quantization.fp8_kernel.deep_gemm_fp8_fp8_bf16_nt`
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

**Peak: 28.6% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 24
- Capture note: Captured on verda-b300-fin-03-1/light-face-hides-fin-03-1, sglang-glm52 container, zai-org/GLM-5.2-FP8 tp=8, CUDA graphs disabled for Python API capture, 2026-07-07. Final records include DSA/TRTLLM attention, torch.bmm, DeepGEMM, and per-token quant call contracts from real server requests.

Functions covered:
- `sglang.srt.layers.quantization.fp8_kernel.deep_gemm_fp8_fp8_bf16_nt`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path zai-org/GLM-5.2-FP8 --tp 8 --trust-remote-code --mem-fraction-static 0.8
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
