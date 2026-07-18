# Profile evidence - intern_s2_preview__linear_gemm

**Standalone kernel target: 42.0% of total serving GPU time** (max across scenarios) on
`internlm/Intern-S2-Preview`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `internlm/Intern-S2-Preview` production-path capture and replace the old noisy profiler shape strings.

- Model: `internlm/Intern-S2-Preview` (slug `intern_s2_preview`, tp=8)
- Python interface(s): `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_16x64_64x16_4x1_v_bz_TNN`, `nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN`, `nvjet_sm100_tst_64x16_64x16_2x4_2cta_h_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_1x2_h_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_TNT`, `nvjet_sm100_tst_64x8_64x16_2x4_h_bz_splitK_TNT`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 41.99% |
| random | conc 32 | 17.59% |
| random | conc 100 | 5.37% |
| sharegpt | conc 1 | 41.84% |
| sharegpt | conc 32 | 24.46% |
| sharegpt | conc 100 | 21.69% |

**Peak: 42.0% in `random_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 18
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real internlm/Intern-S2-Preview TP=8 SGLang production-path execution in temporary container sglang-intern-s2-preview on eight B300 GPUs. Used a model-local HF cache after snapshot download, trust_remote_code=True, reasoning_parser=qwen3, tool_call_parser=qwen3_coder, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, cleared startup health records before capture, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency. The legacy intern_s2_preview__fp8_bmm task has no standalone torch.bmm/FP8 BMM Python API in this capture; it is routed to the real Triton attention backend APIs emitted by the production path rather than synthetic BMM shapes.

Functions covered:
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
