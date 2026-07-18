# Profile evidence - glm_5__linear_gemm

**Standalone kernel target: 33.9% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-5-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `nvidia/GLM-5-NVFP4` production-path capture and replace the old noisy profiler shape strings.

- Model: `nvidia/GLM-5-NVFP4` (slug `glm_5`, tp=4)
- Python interface(s): `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_192x288_64x5_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_256x128_64x5_2x2_2cta_h_bz_TNT`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x16_64x16_2x4_2cta_h_bz_splitK_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_splitK_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 32.24% |
| random | conc 32 | 14.22% |
| random | conc 100 | 18.55% |
| sharegpt | conc 1 | 33.85% |
| sharegpt | conc 32 | 15.95% |
| sharegpt | conc 100 | 15.28% |

**Peak: 33.9% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 12
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real nvidia/GLM-5-NVFP4 TP=4 SGLang production-path execution in temporary container sglang-glm5. Used a model-local HF cache after snapshot download, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, attention_backend=dsa with TRTLLM DSA prefill/decode as selected by SGLang, moe_runner_backend=flashinfer_trtllm, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, and marked five request windows covering long prefill, short decode, mid concurrency, high concurrency, and an extra-long DSA/indexer prompt. The legacy fast_hadamard task maps to the captured DSA backend forward APIs because the current cookbook path did not emit a standalone hadamard_transform Python API; the old fast_hadamard GPU work is fused inside the DSA path.

Functions covered:
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
