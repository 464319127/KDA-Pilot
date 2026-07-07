# Profile evidence - glm_52__per_token_group_quant

**Standalone kernel target: 10.9% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-5.2-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `zai-org/GLM-5.2-FP8` production-path capture and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-5.2-FP8` (slug `glm_52`, tp=8)
- Python interface(s): `sglang.srt.layers.quantization.fp8_kernel.per_token_group_quant_fp8`, `sglang.srt.layers.quantization.fp8_kernel.sglang_per_token_group_quant_fp8`
- Kernel family: `per_token_group_quant`  .  Category: `quant_gemm`
- GPU kernel(s): `void (anonymous namespace)::per_token_group_quant_8bit_v2_kernel<(anonymous namespace)::Na`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 10.92% |
| random | conc 100 | 5.25% |
| sharegpt | conc 1 | 10.74% |
| sharegpt | conc 32 | 5.52% |
| sharegpt | conc 100 | 8.78% |

**Peak: 10.9% in `random_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 19
- Capture note: Captured on verda-b300-fin-03-1/light-face-hides-fin-03-1, sglang-glm52 container, zai-org/GLM-5.2-FP8 tp=8, CUDA graphs disabled for Python API capture, 2026-07-07. Final records include DSA/TRTLLM attention, torch.bmm, DeepGEMM, and per-token quant call contracts from real server requests.

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
