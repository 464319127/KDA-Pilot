# Profile evidence - lfm25__sgl_kernel_moe_align_block_size

**Standalone kernel target: 5.1% of total serving GPU time** (max across scenarios) on
`LiquidAI/LFM2.5-8B-A1B`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `LiquidAI/LFM2.5-8B-A1B` production-path capture and replace the old noisy profiler shape strings.

- Model: `LiquidAI/LFM2.5-8B-A1B` (slug `lfm25`, tp=1)
- Python interface(s): `sgl_kernel.moe_align_block_size`, `sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size.moe_align_block_size`
- Kernel family: `None`  .  Category: `moe`
- GPU kernel(s): `void moe_align_block_size_small_batch_expert_kernel<int, 256>(int const*, int*, int*, int*`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 5.09% |

**Peak: 5.1% in `random_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 20
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real LiquidAI/LFM2.5-8B-A1B TP=1 SGLang production-path execution in temporary container sglang-lfm25 on a single B300 GPU. Used a model-local HF cache after snapshot download, attention_backend=flashinfer, reasoning_parser=qwen3, tool_call_parser=lfm2, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `sgl_kernel.moe_align_block_size`
- `sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size.moe_align_block_size`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
