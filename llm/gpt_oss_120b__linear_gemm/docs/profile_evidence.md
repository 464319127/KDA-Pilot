# Profile evidence - gpt_oss_120b__linear_gemm

**Standalone kernel target: 29.8% of total serving GPU time** (max across scenarios) on
`openai/gpt-oss-120b`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `openai/gpt-oss-120b` production-path capture and replace the old noisy profiler shape strings.

- Model: `openai/gpt-oss-120b` (slug `gpt_oss_120b`, tp=8)
- Python interface(s): `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_24x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_bias_TNN`, `nvjet_sm100_tst_32x64_64x16_4x2_2cta_h_bz_splitK_bias_TNN`, `nvjet_sm100_tst_64x16_64x16_2x4_2cta_h_bz_splitK_bias_TNT`, `nvjet_sm100_tst_64x32_64x16_2x4_2cta_h_bz_splitK_bias_TNT`, `nvjet_sm100_tst_64x8_64x16_1x2_h_bz_bias_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_bias_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_splitK_bias_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 27.53% |
| random | conc 32 | 9.67% |
| random | conc 100 | 26.43% |
| sharegpt | conc 1 | 29.79% |
| sharegpt | conc 32 | 17.52% |
| sharegpt | conc 100 | 20.69% |

**Peak: 29.8% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 32
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real openai/gpt-oss-120b TP=8 SGLang production-path execution in temporary container sglang-gpt-oss-120b on eight B300 GPUs. Used a model-local HF cache after snapshot download, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy gpt_oss_120b__fp8_bmm task is routed to the real low-level FlashInfer TRTLLM batch context/decode APIs observed in the production path, while the attention task uses the higher-level TRTLLM backend forward APIs.

Functions covered:
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
