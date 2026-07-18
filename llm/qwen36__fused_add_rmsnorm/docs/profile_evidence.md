# Profile evidence - qwen36__fused_add_rmsnorm

**Standalone kernel target: 3.5% of total serving GPU time** (max across scenarios) on
`Qwen/Qwen3.6-35B-A3B-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `Qwen/Qwen3.6-35B-A3B-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `Qwen/Qwen3.6-35B-A3B-FP8` (slug `qwen36`, tp=1)
- Python interface(s): `sglang.srt.layers.layernorm.gemma_fused_add_rmsnorm`
- Kernel family: `fused_add_rmsnorm`  .  Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 3.51% |
| sharegpt | conc 1 | 3.53% |

**Peak: 3.5% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 14
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real Qwen/Qwen3.6-35B-A3B-FP8 TP=1 SGLang production-path execution in temporary container sglang-qwen36 on a single B300 GPU. Used a model-local HF cache after snapshot download, speculative_algorithm=EAGLE with 3 steps/topk 1 per the cookbook command, quantization=fp8 as detected by SGLang, attention_backend=trtllm_mha, moe_runner_backend=flashinfer_trtllm, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy fp8_bmm task maps to captured TRTLLM MHA attention APIs because this Qwen3.6 route did not emit a standalone bmm Python API; the old bmm GPU work is fused inside the attention backend.

Functions covered:
- `sglang.srt.layers.layernorm.gemma_fused_add_rmsnorm`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
