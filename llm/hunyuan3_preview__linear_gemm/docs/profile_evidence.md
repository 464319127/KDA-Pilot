# Profile evidence - hunyuan3_preview__linear_gemm

**Standalone kernel target: 8.4% of total serving GPU time** (max across scenarios) on
`tencent/Hy3-preview`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `tencent/Hy3-preview` production-path capture and replace the old noisy profiler shape strings.

- Model: `tencent/Hy3-preview` (slug `hunyuan3_preview`, tp=8)
- Python interface(s): `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `gemm`
- GPU kernel(s): `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x128x16_1x1x1_3_tnn_align1_bias_f32_relu`, `nvjet_hsh_32x64_64x16_4x1_v_bz_splitK_TNN`, `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_tn_align1>(cutlass_80_simt_sgemm_64x`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 6.96% |
| random | conc 32 | 8.42% |
| sharegpt | conc 1 | 5.50% |

**Peak: 8.4% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 22
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real tencent/Hy3-preview TP=8 SGLang production-path execution in temporary container sglang-hunyuan3-preview on eight B300 GPUs. Used a model-local HF cache after snapshot download, cookbook-aligned EAGLE speculative settings (--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1), disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
