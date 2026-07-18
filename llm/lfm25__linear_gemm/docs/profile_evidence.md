# Profile evidence - lfm25__linear_gemm

**Standalone kernel target: 19.0% of total serving GPU time** (max across scenarios) on
`LiquidAI/LFM2.5-8B-A1B`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `LiquidAI/LFM2.5-8B-A1B` production-path capture and replace the old noisy profiler shape strings.

- Model: `LiquidAI/LFM2.5-8B-A1B` (slug `lfm25`, tp=1)
- Python interface(s): `torch.nn.functional.linear`
- Kernel family: `None`  .  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_128x8_64x12_2x1_v_bz_TNT`, `nvjet_sm100_tst_16x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_256x128_64x5_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_1x1_h_bz_splitK_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 19.02% |
| random | conc 32 | 5.08% |
| random | conc 100 | 11.24% |
| sharegpt | conc 1 | 17.51% |
| sharegpt | conc 32 | 5.65% |
| sharegpt | conc 100 | 8.73% |

**Peak: 19.0% in `random_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 60
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real LiquidAI/LFM2.5-8B-A1B TP=1 SGLang production-path execution in temporary container sglang-lfm25 on a single B300 GPU. Used a model-local HF cache after snapshot download, attention_backend=flashinfer, reasoning_parser=qwen3, tool_call_parser=lfm2, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
