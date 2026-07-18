# Profile evidence - qwen3_next__fp8_bmm

**Standalone kernel target: 5.9% of total serving GPU time** (max across scenarios) on
`Qwen/Qwen3-Next-80B-A3B-Instruct`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `Qwen/Qwen3-Next-80B-A3B-Instruct` production-path capture and replace the old noisy profiler shape strings.

- Model: `Qwen/Qwen3-Next-80B-A3B-Instruct` (slug `qwen3_next`, tp=8)
- Python interface(s): `sglang.srt.layers.attention.triton_backend.TritonAttnBackend.forward_decode`, `sglang.srt.layers.attention.triton_backend.TritonAttnBackend.forward_extend`
- Kernel family: `fp8_bmm`  .  Category: `other`
- GPU kernel(s): `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x64x128_s5_et128x64_m256x64x16_c2x1x1_rM_BN_transOu`, `bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x64x128u2_s5_et128x64_m256x64x16_c2x1x1_rM_BN_trans`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 32 | 2.03% |
| random | conc 100 | 5.88% |

**Peak: 5.9% in `random_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 9
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real Qwen/Qwen3-Next-80B-A3B-Instruct TP=8 SGLang production-path execution in temporary container sglang-qwen3-next on eight B300 GPUs. Used a model-local HF cache after snapshot download, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy qwen3_next__fp8_bmm task has no standalone torch.bmm/FP8 BMM Python API in this capture; it is routed to the real Triton attention backend APIs emitted by the production path rather than synthetic BMM shapes.

Functions covered:
- `sglang.srt.layers.attention.triton_backend.TritonAttnBackend.forward_decode`
- `sglang.srt.layers.attention.triton_backend.TritonAttnBackend.forward_extend`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
