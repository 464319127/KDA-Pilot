# Profile evidence - qwen3_coder_next__linear_gemm

**Standalone kernel target: 40.5% of total serving GPU time** (max across scenarios) on
`Qwen/Qwen3-Coder-Next`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `Qwen/Qwen3-Coder-Next` production-path capture and replace the old noisy profiler shape strings.

- Model: `Qwen/Qwen3-Coder-Next` (slug `qwen3_coder_next`, tp=2)
- Python interface(s): `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_128x24_64x11_4x2_h_bz_TNT`, `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_16x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x16_64x16_1x2_h_bz_TNT`, `nvjet_sm100_tst_64x24_64x16_1x2_h_bz_TNT`, `nvjet_sm100_tst_64x24_64x16_4x1_v_bz_TNT`, `nvjet_sm100_tst_64x64_64x16_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`, `nvjet_sm100_tst_8x64_64x16_4x1_v_bz_TNN`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 40.52% |
| random | conc 32 | 12.37% |
| random | conc 100 | 15.02% |
| sharegpt | conc 1 | 27.03% |
| sharegpt | conc 32 | 18.76% |
| sharegpt | conc 100 | 16.59% |

**Peak: 40.5% in `random_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 18
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real Qwen/Qwen3-Coder-Next TP=2 SGLang production-path execution in temporary container sglang-qwen3-coder-next on two B300 GPUs. Used a model-local HF cache after snapshot download, tool_call_parser=qwen3_coder, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy qwen3_coder_next__fp8_bmm task has no standalone torch.bmm/FP8 BMM Python API in this capture; it is routed to the real Triton attention backend APIs emitted by the production path rather than synthetic BMM shapes.

Functions covered:
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
