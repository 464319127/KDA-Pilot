# Profile evidence - glm_47__fp8_bmm

**Standalone kernel target: 19.5% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-4.7-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `nvidia/GLM-4.7-NVFP4` production-path capture and replace the old noisy profiler shape strings.

- Model: `nvidia/GLM-4.7-NVFP4` (slug `glm_47`, tp=8)
- Python interface(s): `flashinfer.gemm.mm_fp4`
- Kernel family: `fp8_bmm`  .  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_rM_TN_tr`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x32x256_s9_et128x32_m128x32x64_c1x1x1_rM_TN_trans`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_r`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x16x512_s5_et128x16_m256x16x64_c2x1x1_rM_T`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512_s3x3x3x3x1x3_et128x32_m256x128x64_c2x1x`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x16x512_s5_et128x16_m256x16x64_c2x1x1_rM_TN_tran`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x16x512u2_s5_et128x16_m128x16x64_c1x1x1_rM_TN_tr`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x32x512_s4_et128x32_m128x32x64_c1x1x1_rM_TN_tran`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x32x512u2_s4_et128x32_m128x32x64_c1x1x1_rM_TN_tr`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512u2_s5_et128x8_m128x8x64_c1x1x1_rM_TN_trans`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 7.35% |
| random | conc 32 | 12.48% |
| random | conc 100 | 19.49% |
| sharegpt | conc 1 | 5.34% |
| sharegpt | conc 32 | 12.80% |
| sharegpt | conc 100 | 18.73% |

**Peak: 19.5% in `random_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 12
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real nvidia/GLM-4.7-NVFP4 TP=8 SGLang production-path execution in temporary container sglang-glm47. Used a model-local HF cache in offline mode after snapshot download, trust_remote_code, quantization=modelopt_fp4, reasoning_parser=glm45, attention_backend=trtllm_mha/FlashInfer default for this NVFP4 checkpoint, moe_runner_backend=flashinfer_trtllm as selected by SGLang for ModelOpt NVFP4 MoE, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy fp8_bmm task maps to the captured FlashInfer NVFP4 mm_fp4 Python API because the old profiler class name did not correspond to a bmm_fp8 Python call in this production path.

Functions covered:
- `flashinfer.gemm.mm_fp4`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
