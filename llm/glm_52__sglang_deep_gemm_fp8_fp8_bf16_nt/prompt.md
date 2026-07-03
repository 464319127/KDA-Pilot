# KDA Prompt: glm_52__sglang_deep_gemm_fp8_fp8_bf16_nt

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `sglang.deep_gemm_fp8_fp8_bf16_nt`

## Environment And Runner

Use the `ion-b200` remote GPU environment for all B200 work. All CUDA, Python,
pip, nvcc, build, test, benchmark, and profiling commands must run inside the
existing `sglang_bbuf` Docker container on `ion-b200`.

Before GPU work, inspect `nvidia-smi` and choose a B200 GPU with no active
compute processes and no meaningful memory occupancy. Export that id as
`REMOTE_GPU_ID` and use it consistently for baseline, candidate, benchmark,
profiler, and NCU commands in the current run. Do not run measurements on busy
cards or directly on the `ion-b200` host.

Launch this task with `CLAUDE_MODEL=fable5`.

**28.6% of total GPU time** on `zai-org/GLM-5.2-FP8` (cookbook-aligned profile, peak
`sharegpt_low`) — a genuine end-to-end target selected by profiler e2e share. Family
`linear_gemm`, category `quant_gemm`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
the cookbook deployment, and the scenario to re-run for the e2e A/B. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
