# KDA Prompt: glm_52__fp8_bmm

Target GPU: NVIDIA B200. Optimize the SGLang kernel behind:

- `<confirm via capture; profiler family=fp8_bmm>`

## Environment And Runner

Use the `ion-b200` remote GPU environment for all B200 work. All CUDA, Python,
pip, nvcc, build, test, benchmark, and profiling commands must run inside the
existing `sglang_bbuf` Docker container on `ion-b200`.

Before GPU work, inspect `nvidia-smi` and choose a B200 GPU with no active
compute processes and no meaningful memory occupancy. Export that id as
`REMOTE_GPU_ID` and use it consistently for baseline, candidate, benchmark,
profiler, and NCU commands in the current run. Do not run measurements on busy
cards or directly on the `ion-b200` host.

**25.0% of total serving GPU time** on `zai-org/GLM-5.2-FP8` (cookbook-aligned profile, peak
`sharegpt_mid`) — a serving-profile headroom signal used to select this standalone kernel task. Family
`fp8_bmm`, category `quant_gemm`.

See `docs/profile_evidence.md` for the per-scenario %-of-GPU, GPU kernels, shapes,
and original serving capture provenance. Do not start/re-run SGLang serve,
`run_capture`, or a multi-GPU e2e A/B for the normal RLCR loop; optimize and
validate via the task-local standalone benchmark on one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
