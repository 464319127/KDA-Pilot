# Profile evidence - kimi_k2__fp8_bmm

**Standalone kernel target: 38.5% of total serving GPU time** (max across scenarios) on
`moonshotai/Kimi-K2-Instruct`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below were recaptured from a real `moonshotai/Kimi-K2-Instruct` server run and replace the old noisy profiler shape strings.

- Model: `moonshotai/Kimi-K2-Instruct` (slug `kimi_k2`, tp=8)
- Python interface(s): `torch.bmm`
- Kernel family: `fp8_bmm`  .  Category: `quant_gemm`
- GPU kernel(s): `bmm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_TN_transOut_noShfl`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x32x128_s6_et64x32_m64x32x32_c1x1x1_rM_TN_transOut_noShfl_d`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_c1x1x1_rM_TN_transOut_noShfl`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s6_et64x64_m64x64x32_c1x1x1_rM_TN_transOut_noShfl_d`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128u2_s6_et64x64_m64x64x32_c1x1x1_rM_TN_transOut_noShfl`, `bmm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_TN_transOut_noShfl_ds`, `bmm_E4m3_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_rM_TN_transOut_noShfl_dsF`, `bmm_E4m3_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_c1x1x1_rM_TN_transOut_noShfl_dsF`, `bmm_E4m3_E4m3E4m3_Fp32_t128x64x128u2_s6_et64x64_m64x64x32_c1x1x1_rM_TN_transOut_noShfl_dsF`, `bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_c1x1x1_rM_TN_transOut_noShfl_dsFp8_`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 13.16% |
| random | conc 32 | 6.77% |
| random | conc 100 | 21.20% |
| sharegpt | conc 1 | 10.91% |
| sharegpt | conc 32 | 14.13% |
| sharegpt | conc 100 | 38.52% |

**Peak: 38.5% in `sharegpt_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 18
- Capture note: Captured on 2026-07-07 from a real moonshotai/Kimi-K2-Instruct SGLang TP=8 server on Verda B300 node light-face-hides-fin-03-1, container sglang-kimi-k2-local, local NVMe Hugging Face cache, HF offline mode, trust_remote_code, tool_call_parser=kimi_k2, cuda graph prefill/decode disabled for Python API capture. Runtime selected attention_backend=trtllm_mla and moe_runner_backend=flashinfer_trtllm(auto). Records include server startup/JIT/autotune API calls plus marked request windows for sharegpt_low_long_prompt, random_low_short_prompt, sharegpt_mid_concurrency_long_prompt, and random_high_concurrency_short_prompt.

Functions covered:
- `torch.bmm`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Original serving profile command (provenance only)
```bash
sglang serve --model-path moonshotai/Kimi-K2-Instruct --tp 8 --tool-call-parser kimi_k2
```
This command is retained only to explain target selection. Normal RLCR kernel
work must not depend on a live SGLang server, `run_capture`, 8-GPU availability,
or a multi-GPU e2e gate. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set. Re-run serving capture only
when intentionally refreshing these evidence files.
