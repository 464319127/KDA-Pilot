# Profile evidence - glm_46__linear_gemm

**Standalone kernel target: 7.1% of total serving GPU time** (max across scenarios) on
`zai-org/GLM-4.6-FP8`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below were recaptured from a real `zai-org/GLM-4.6-FP8` server run and replace the old noisy profiler shape strings.

- Model: `zai-org/GLM-4.6-FP8` (slug `glm_46`, tp=8)
- Python interface(s): `sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear`, `sglang.srt.layers.quantization.fp8_utils.fp8_scaled_mm`, `torch.nn.functional.linear`
- Kernel family: `linear_gemm`  .  Category: `gemm`
- GPU kernel(s): `_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEENS1_10colle`, `cutlass3x_sm100_tensorop_s128x64x8tf32gemm_f32_f32_f32_f32_f32_128x64x32_0_tnn_align4_2sm_`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 7.10% |
| random | conc 32 | 2.27% |
| random | conc 100 | 2.70% |
| sharegpt | conc 1 | 6.87% |
| sharegpt | conc 100 | 3.03% |

**Peak: 7.1% in `random_low`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 27
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real zai-org/GLM-4.6-FP8 SGLang TP=8 server in temporary container sglang-glm46. Used a model-local HF cache in offline mode after snapshot download, trust_remote_code, reasoning_parser=glm45, tool_call_parser=glm45 (server_args normalized it to glm; chat template auto-detected glm45), attention_backend=fa4, moe_runner_backend=triton for the Triton MoE tasks, CUDA graph prefill/decode disabled, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear`
- `sglang.srt.layers.quantization.fp8_utils.fp8_scaled_mm`
- `torch.nn.functional.linear`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Original serving profile command (provenance only)
```bash
sglang serve --model-path zai-org/GLM-4.6-FP8 --tp 8 --reasoning-parser glm45 --tool-call-parser glm45 --attention-backend fa4
```
This command is retained only to explain target selection. Normal RLCR kernel
work must not depend on a live SGLang server, `run_capture`, 8-GPU availability,
or a multi-GPU e2e gate. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set. Re-run serving capture only
when intentionally refreshing these evidence files.
