# Profile evidence - ernie45__linear_gemm

**Standalone kernel target: 29.1% of total serving GPU time** (max across scenarios) on
`baidu/ERNIE-4.5-21B-A3B-PT`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `baidu/ERNIE-4.5-21B-A3B-PT` production-path capture and replace the old noisy profiler shape strings.

- Model: `baidu/ERNIE-4.5-21B-A3B-PT` (slug `ernie45`, tp=1)
- Python interface(s): `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `nvjet_sm100_tst_128x256_64x6_2x2_2cta_h_bz_TNT`, `nvjet_sm100_tst_32x64_64x16_2x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_64x64_64x16_2x2_2cta_h_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_2x1_v_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_4x1_v_bz_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 27.31% |
| random | conc 32 | 13.54% |
| random | conc 100 | 2.90% |
| sharegpt | conc 1 | 29.07% |
| sharegpt | conc 32 | 9.63% |
| sharegpt | conc 100 | 2.55% |

**Peak: 29.1% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 42
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real baidu/ERNIE-4.5-21B-A3B-PT SGLang TP=1 server in container sglang-ernie45. Used local NVMe HF cache in offline mode, trust_remote_code, attention_backend=trtllm_mha, moe_runner_backend=triton for the Triton MoE task, CUDA graph prefill/decode disabled, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
