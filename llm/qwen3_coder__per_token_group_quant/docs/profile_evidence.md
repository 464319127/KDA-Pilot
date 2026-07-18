# Profile evidence - qwen3_coder__per_token_group_quant

**Standalone kernel target: 17.2% of total serving GPU time** (max across scenarios) on
`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` (slug `qwen3_coder`, tp=8)
- Python interface(s): `sglang.srt.layers.quantization.fp8_kernel.per_token_group_quant_fp8`, `sglang.srt.layers.quantization.fp8_kernel.sglang_per_token_group_quant_fp8`
- Kernel family: `per_token_group_quant`  .  Category: `quant_gemm`
- GPU kernel(s): `void per_token_group_quant_8bit_kernel<NaiveScheduler, 128, 8, __nv_bfloat16, c10::Float8_`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 17.25% |
| random | conc 100 | 8.00% |
| sharegpt | conc 1 | 16.25% |
| sharegpt | conc 32 | 7.58% |

**Peak: 17.2% in `random_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 21
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 TP=8 EP=8 SGLang production-path execution in temporary container sglang-qwen3-coder on eight B300 GPUs. Used a model-local HF cache after snapshot download, tool_call_parser=qwen3_coder, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy qwen3_coder__fp8_bmm task is routed to the real low-level FlashInfer TRTLLM batch context/decode APIs observed in the production path, while the attention task uses the higher-level TRTLLM backend forward APIs.

Functions covered:
- `sglang.srt.layers.quantization.fp8_kernel.per_token_group_quant_fp8`
- `sglang.srt.layers.quantization.fp8_kernel.sglang_per_token_group_quant_fp8`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
