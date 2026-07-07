# Profile evidence - ernie45__fused_moe_triton

**Standalone kernel target: 65.0% of total serving GPU time** (max across scenarios) on
`baidu/ERNIE-4.5-21B-A3B-PT`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below were recaptured from a real `baidu/ERNIE-4.5-21B-A3B-PT` server run and replace the old noisy profiler shape strings.

- Model: `baidu/ERNIE-4.5-21B-A3B-PT` (slug `ernie45`, tp=1)
- Python interface(s): `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._fused_moe_kernel_sequence`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts_impl`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts`, `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels.invoke_fused_moe_kernel`
- Kernel family: `fused_moe_triton`  .  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 37.00% |
| random | conc 32 | 63.38% |
| random | conc 100 | 64.96% |
| sharegpt | conc 1 | 35.86% |
| sharegpt | conc 32 | 62.86% |
| sharegpt | conc 100 | 64.82% |

**Peak: 65.0% in `random_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 288
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real baidu/ERNIE-4.5-21B-A3B-PT SGLang TP=1 server in container sglang-ernie45. Used local NVMe HF cache in offline mode, trust_remote_code, attention_backend=trtllm_mha, moe_runner_backend=triton for the Triton MoE task, CUDA graph prefill/decode disabled, and marked four request windows covering long prefill, short decode, mid concurrency, and high concurrency.

Functions covered:
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe._fused_moe_kernel_sequence`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts_impl`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts`
- `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels.invoke_fused_moe_kernel`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Original serving profile command (provenance only)
```bash
sglang serve --model-path baidu/ERNIE-4.5-21B-A3B-PT --tp 1
```
This command is retained only to explain target selection. Normal RLCR kernel
work must not depend on a live SGLang server, `run_capture`, 8-GPU availability,
or a multi-GPU e2e gate. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set. Re-run serving capture only
when intentionally refreshing these evidence files.
