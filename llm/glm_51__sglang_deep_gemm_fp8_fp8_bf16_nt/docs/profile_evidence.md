# Profile evidence - glm_51__sglang_deep_gemm_fp8_fp8_bf16_nt

**Standalone kernel target: 20.6% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-5.1-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `zai-org/GLM-5.1-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-5.1-FP8` (slug `glm_51`, tp=8)
- Python interface(s): `sglang.srt.layers.quantization.fp8_kernel.deep_gemm_fp8_fp8_bf16_nt`
- Kernel family: `linear_gemm`  .  Category: `quant_gemm`
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

**Peak: 20.6% in `random_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 17
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real zai-org/GLM-5.1-FP8 TP=8 SGLang production-path execution in temporary container sglang-glm51. Used a model-local HF cache after snapshot download, quantization=fp8 as detected by SGLang, kv_cache_dtype=fp8_e4m3, attention_backend=dsa with TRTLLM DSA prefill/decode, moe_runner_backend=flashinfer_trtllm, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The generic attention task maps to the captured DSA backend forward APIs because this GLM-5.1 route uses DSA/TRTLLM MLA attention rather than a standalone generic attention Python API.

Functions covered:
- `sglang.srt.layers.quantization.fp8_kernel.deep_gemm_fp8_fp8_bf16_nt`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
