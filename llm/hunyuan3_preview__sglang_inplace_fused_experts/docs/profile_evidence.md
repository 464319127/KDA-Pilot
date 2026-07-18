# Profile evidence - hunyuan3_preview__sglang_inplace_fused_experts

**Standalone kernel target: 34.7% of total serving GPU time** (max across scenarios) on
`tencent/Hy3-preview`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `tencent/Hy3-preview` production-path capture and replace the old noisy profiler shape strings.

- Model: `tencent/Hy3-preview` (slug `hunyuan3_preview`, tp=8)
- Python interface(s): `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._fused_moe_kernel_sequence`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts_impl`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels.invoke_fused_moe_kernel`
- Kernel family: `fused_moe_triton`  .  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 15.84% |
| random | conc 32 | 34.73% |
| random | conc 100 | 20.95% |
| sharegpt | conc 1 | 12.37% |
| sharegpt | conc 32 | 21.16% |
| sharegpt | conc 100 | 18.92% |

**Peak: 34.7% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 73
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real tencent/Hy3-preview TP=8 SGLang production-path execution in temporary container sglang-hunyuan3-preview on eight B300 GPUs. Used a model-local HF cache after snapshot download, cookbook-aligned EAGLE speculative settings (--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1), disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._fused_moe_kernel_sequence`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts_impl`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels.invoke_fused_moe_kernel`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
