# Profile evidence - glm_47_flash__sglang_inplace_fused_experts

**Standalone kernel target: 30.4% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-4.7-Flash`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `zai-org/GLM-4.7-Flash` production-path capture and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-4.7-Flash` (slug `glm_47_flash`, tp=1)
- Python interface(s): `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._fused_moe_kernel_sequence`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts_impl`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels.invoke_fused_moe_kernel`
- Kernel family: `fused_moe_triton`  .  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 21.80% |
| random | conc 32 | 30.39% |
| random | conc 100 | 27.34% |
| sharegpt | conc 1 | 20.23% |
| sharegpt | conc 32 | 12.17% |
| sharegpt | conc 100 | 29.47% |

**Peak: 30.4% in `random_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 35
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real zai-org/GLM-4.7-Flash TP=1 SGLang production-path execution in temporary container sglang-glm47-flash. Used a model-local HF cache after snapshot download, attention_backend=triton, reasoning_parser=glm45, tool_call_parser=glm47, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The attention task maps to the captured TritonAttnBackend forward APIs because this Flash path did not call the older unified_attention_with_output Python wrapper during the capture.

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
