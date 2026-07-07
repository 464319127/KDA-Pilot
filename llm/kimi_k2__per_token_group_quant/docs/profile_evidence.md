# Profile evidence - kimi_k2__per_token_group_quant

**Standalone kernel target: 12.6% of total serving GPU time** (max across scenarios) on
`moonshotai/Kimi-K2-Instruct`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below were recaptured from a real `moonshotai/Kimi-K2-Instruct` server run and replace the old noisy profiler shape strings.

- Model: `moonshotai/Kimi-K2-Instruct` (slug `kimi_k2`, tp=8)
- Python interface(s): `sglang.srt.layers.quantization.fp8_kernel.per_token_group_quant_fp8`, `sglang.srt.layers.quantization.fp8_kernel.sglang_per_token_group_quant_fp8`
- Kernel family: `per_token_group_quant`  .  Category: `quant_gemm`
- GPU kernel(s): `void per_token_group_quant_8bit_kernel<NaiveScheduler, 128, 8, __nv_bfloat16, c10::Float8_`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 12.64% |
| random | conc 32 | 2.42% |
| random | conc 100 | 6.24% |
| sharegpt | conc 1 | 12.59% |
| sharegpt | conc 32 | 6.30% |
| sharegpt | conc 100 | 6.16% |

**Peak: 12.6% in `random_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 8
- Capture note: Captured on 2026-07-07 from a real moonshotai/Kimi-K2-Instruct SGLang TP=8 server on Verda B300 node light-face-hides-fin-03-1, container sglang-kimi-k2-local, local NVMe Hugging Face cache, HF offline mode, trust_remote_code, tool_call_parser=kimi_k2, cuda graph prefill/decode disabled for Python API capture. Runtime selected attention_backend=trtllm_mla and moe_runner_backend=flashinfer_trtllm(auto). Records include server startup/JIT/autotune API calls plus marked request windows for sharegpt_low_long_prompt, random_low_short_prompt, sharegpt_mid_concurrency_long_prompt, and random_high_concurrency_short_prompt.

Functions covered:
- `sglang.srt.layers.quantization.fp8_kernel.per_token_group_quant_fp8`
- `sglang.srt.layers.quantization.fp8_kernel.sglang_per_token_group_quant_fp8`

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
