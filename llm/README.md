# KDA-Pilot — LLM Kernel-Workflow Campaign

This is the `llm/` subtree of KDA-Pilot. For each priority autoregressive model
we serve it on B200 and H200 using the **sgl-cookbook** deployment command,
benchmark it at low / mid / high concurrency with the cookbook's dataset method,
profile the serving forward pass, and turn every kernel that takes **≥1% of GPU
time** (excluding attention and cuDNN) into a kernel-optimization task card —
mirroring the `diffusion/` task format.

## Goal per model

1. Deploy via the exact cookbook `sglang serve` command for the platform.
2. Run `sglang.bench_serving` at low / mid / high `--max-concurrency`.
3. Capture a torch profiler trace of the forward pass under load.
4. Extract a **kernel-workflow inventory**: every ≥1% kernel, categorized
   (gemm / quant-gemm / moe / norm / rope / memory-bound / comm / …), with
   attention + cuDNN kernels reported as an aggregate but **not** turned into
   tasks.
5. Each ≥1% non-attention kernel becomes a per-kernel optimization task card.
6. Commit the model's folder, then **delete the model weights from the remote
   box** to protect disk.

## Layout

```text
docs/
  llm_kernel_workflow_rules.md   # what to record, the ≥1% rule, categories, exclusions
  model_priority.md              # priority list + status tracker
scripts/
  serve.sh                       # start `sglang serve`, wait for readiness
  bench.sh                       # bench_serving sweep: low / mid / high concurrency
  profile_forward.sh             # capture a torch profiler trace via the server profiler
  extract_kernel_workflow.py     # trace.json[.gz] -> kernel_workflow.{md,csv}
<model>/<platform>/              # e.g. minimax_m27/b200/
  deploy.md                      # exact serve + bench + profile commands used
  run_log.md                     # provenance: host, GPU ids, image, commit, idle state
  bench/                         # bench_serving logs per concurrency level
  profile/                       # raw + parsed profiler artifacts
  docs/kernel_workflow.md        # the ≥1% kernel inventory (the headline deliverable)
  kernels/<kernel-task>/         # per-kernel optimization task cards (diffusion-style)
```

`<model>` slugs are lowercase, `.`/`-` → `_` (e.g. `MiniMax-M2.7` → `minimax_m27`).
`<platform>` is `b200` or `h200`.

## Running (on the remote box)

Clone this repo inside the SGLang dev container and drive the scripts from
`llm/`. See each `<model>/<platform>/deploy.md` for the concrete commands and
`docs/llm_kernel_workflow_rules.md` for the recording contract.
