# Profile evidence - glm_5__fp8_bmm

**Standalone kernel target: 19.5% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-5-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `nvidia/GLM-5-NVFP4` production-path capture and replace the old noisy profiler shape strings.

- Model: `nvidia/GLM-5-NVFP4` (slug `glm_5`, tp=4)
- Python interface(s): `flashinfer.gemm.mm_fp4`, `torch.bmm`
- Kernel family: `fp8_bmm`  .  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x128x256u2_s6_et128x128_m256x128x64_c2x1x1_rM_TN_`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOut`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x16x512u2_s5_et128x16_m128x16x64_c1x1x1_rM_TN_tr`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOu`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512u2_s5_et128x8_m128x8x64_c1x1x1_rM_TN_trans`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.50% |
| random | conc 32 | 19.50% |
| random | conc 100 | 17.70% |
| sharegpt | conc 1 | 5.25% |
| sharegpt | conc 32 | 14.88% |
| sharegpt | conc 100 | 11.56% |

**Peak: 19.5% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 49
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real nvidia/GLM-5-NVFP4 TP=4 SGLang production-path execution in temporary container sglang-glm5. Used a model-local HF cache after snapshot download, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, attention_backend=dsa with TRTLLM DSA prefill/decode as selected by SGLang, moe_runner_backend=flashinfer_trtllm, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, and marked five request windows covering long prefill, short decode, mid concurrency, high concurrency, and an extra-long DSA/indexer prompt. The legacy fast_hadamard task maps to the captured DSA backend forward APIs because the current cookbook path did not emit a standalone hadamard_transform Python API; the old fast_hadamard GPU work is fused inside the DSA path.

Functions covered:
- `flashinfer.gemm.mm_fp4`
- `torch.bmm`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
