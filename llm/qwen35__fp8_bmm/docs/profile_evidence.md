# Profile evidence - qwen35__fp8_bmm

**Standalone kernel target: 16.0% of total serving GPU time** (max across scenarios) on
`nvidia/Qwen3.5-397B-A17B-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `nvidia/Qwen3.5-397B-A17B-NVFP4` production-path capture and replace the old noisy profiler shape strings.

- Model: `nvidia/Qwen3.5-397B-A17B-NVFP4` (slug `qwen35`, tp=4)
- Python interface(s): `sglang.srt.layers.attention.triton_backend.TritonAttnBackend.forward_decode`, `sglang.srt.layers.attention.triton_backend.TritonAttnBackend.forward_extend`
- Kernel family: `fp8_bmm`  .  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x8x256_s9_et128x8_m128x8x64_c1x1x1_rM_TN_transOut`, `bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_t128x128x256_s6_et128x128_m256x128x64_c2x1x1_r`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512u2_s3x3x3x3x1x3_et128x32_m256x128x64_c2x`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_transOu`, `bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512u2_s5_et128x8_m128x8x64_c1x1x1_rM_TN_trans`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 15.98% |
| random | conc 100 | 13.66% |
| sharegpt | conc 32 | 2.19% |
| sharegpt | conc 100 | 5.21% |

**Peak: 16.0% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 9
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real nvidia/Qwen3.5-397B-A17B-NVFP4 TP=4 SGLang production-path execution in temporary container sglang-qwen35 on four B300 GPUs. Used a model-local HF cache after snapshot download, reasoning_parser=qwen3, tool_call_parser=qwen3_coder, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy qwen35__fp8_bmm task has no standalone torch.bmm/FP8 BMM Python API in this capture; it is routed to the real Triton attention backend APIs emitted by the production path rather than synthetic BMM shapes.

Functions covered:
- `sglang.srt.layers.attention.triton_backend.TritonAttnBackend.forward_decode`
- `sglang.srt.layers.attention.triton_backend.TritonAttnBackend.forward_extend`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
