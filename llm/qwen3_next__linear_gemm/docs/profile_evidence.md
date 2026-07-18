# Profile evidence - qwen3_next__linear_gemm

**Standalone kernel target: 44.3% of total serving GPU time** (max across scenarios) on
`Qwen/Qwen3-Next-80B-A3B-Instruct`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `Qwen/Qwen3-Next-80B-A3B-Instruct` production-path capture and replace the old noisy profiler shape strings.

- Model: `Qwen/Qwen3-Next-80B-A3B-Instruct` (slug `qwen3_next`, tp=8)
- Python interface(s): `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_16x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x8_64x16_1x2_h_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_splitK_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 41.20% |
| random | conc 32 | 16.21% |
| random | conc 100 | 29.40% |
| sharegpt | conc 1 | 44.27% |
| sharegpt | conc 32 | 22.66% |
| sharegpt | conc 100 | 18.96% |

**Peak: 44.3% in `sharegpt_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 18
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real Qwen/Qwen3-Next-80B-A3B-Instruct TP=8 SGLang production-path execution in temporary container sglang-qwen3-next on eight B300 GPUs. Used a model-local HF cache after snapshot download, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy qwen3_next__fp8_bmm task has no standalone torch.bmm/FP8 BMM Python API in this capture; it is routed to the real Triton attention backend APIs emitted by the production path rather than synthetic BMM shapes.

Functions covered:
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
