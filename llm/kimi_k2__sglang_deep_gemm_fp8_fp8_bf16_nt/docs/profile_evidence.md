# Profile evidence - kimi_k2__sglang_deep_gemm_fp8_fp8_bf16_nt

**Standalone kernel target: 23.1% of total serving GPU time** (max across scenarios) on
`moonshotai/Kimi-K2-Instruct`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `moonshotai/Kimi-K2-Instruct` production-path capture and replace the old noisy profiler shape strings.

- Model: `moonshotai/Kimi-K2-Instruct` (slug `kimi_k2`, tp=8)
- Python interface(s): `sglang.srt.layers.quantization.fp8_kernel.deep_gemm_fp8_fp8_bf16_nt`
- Kernel family: `linear_gemm`  .  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_8x64_64x16_4x1_v_bz_TNN`, `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 1`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 18.82% |
| random | conc 32 | 13.60% |
| sharegpt | conc 1 | 23.12% |
| sharegpt | conc 32 | 3.20% |
| sharegpt | conc 100 | 2.23% |

**Peak: 23.1% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 7
- Capture note: Captured on 2026-07-07 from a real moonshotai/Kimi-K2-Instruct SGLang TP=8 server on Verda B300 node light-face-hides-fin-03-1, container sglang-kimi-k2-local, local NVMe Hugging Face cache, HF offline mode, trust_remote_code, tool_call_parser=kimi_k2, cuda graph prefill/decode disabled for Python API capture. Runtime selected attention_backend=trtllm_mla and moe_runner_backend=flashinfer_trtllm(auto). Records include server startup/JIT/autotune API calls plus marked request windows for sharegpt_low_long_prompt, random_low_short_prompt, sharegpt_mid_concurrency_long_prompt, and random_high_concurrency_short_prompt.

Functions covered:
- `sglang.srt.layers.quantization.fp8_kernel.deep_gemm_fp8_fp8_bf16_nt`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
