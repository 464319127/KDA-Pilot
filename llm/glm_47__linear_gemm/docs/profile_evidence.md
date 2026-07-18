# Profile evidence - glm_47__linear_gemm

**Standalone kernel target: 16.5% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-4.7-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `nvidia/GLM-4.7-NVFP4` production-path capture and replace the old noisy profiler shape strings.

- Model: `nvidia/GLM-4.7-NVFP4` (slug `glm_47`, tp=8)
- Python interface(s): `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `quant_gemm`
- GPU kernel(s): `cutlass3x_sm100_tensorop_s128x64x8tf32gemm_f32_f32_f32_f32_f32_128x64x32_0_tnn_align4_2sm_`, `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPer`, `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_128x256_64x6_2x2_2cta_h_bz_bias_TNT`, `nvjet_sm100_tst_128x256_64x6_2x4_2cta_h_bz_bias_TNT`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_bias_TNN`, `nvjet_sm100_tst_64x32_64x16_2x4_2cta_h_bz_bias_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_splitK_bias_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 16.32% |
| random | conc 32 | 9.05% |
| random | conc 100 | 9.97% |
| sharegpt | conc 1 | 16.48% |
| sharegpt | conc 32 | 9.67% |
| sharegpt | conc 100 | 8.63% |

**Peak: 16.5% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 9
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real nvidia/GLM-4.7-NVFP4 TP=8 SGLang production-path execution in temporary container sglang-glm47. Used a model-local HF cache in offline mode after snapshot download, trust_remote_code, quantization=modelopt_fp4, reasoning_parser=glm45, attention_backend=trtllm_mha/FlashInfer default for this NVFP4 checkpoint, moe_runner_backend=flashinfer_trtllm as selected by SGLang for ModelOpt NVFP4 MoE, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy fp8_bmm task maps to the captured FlashInfer NVFP4 mm_fp4 Python API because the old profiler class name did not correspond to a bmm_fp8 Python call in this production path.

Functions covered:
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
