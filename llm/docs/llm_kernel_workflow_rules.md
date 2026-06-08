# LLM Kernel-Workflow Recording Rules

These rules define what each `llm/<model>/<platform>/` folder must capture.

## 1. Deployment

- Use the **exact** sgl-cookbook `sglang serve` command for the model+platform.
  Record the cookbook source (doc path + commit) in `run_log.md`.
- NVIDIA 8-GPU boxes: follow the cookbook hardware table (e.g. MiniMax-M2.7
  B200/H200 = `--tp 8 --ep 8`). Do not invent flags the cookbook does not use.
- Record the docker image, SGLang version/commit, host, GPU ids, and pre/post
  GPU idle state. Benchmarks measured on a non-idle GPU are invalid.

## 2. Benchmark (dataset method from the cookbook)

- Use the dataset method the cookbook's Speed Benchmark section specifies for
  that model (for MiniMax-M2.7: `--dataset-name random --random-input-len 1000
  --random-output-len 1000`).
- Sweep concurrency at three levels — **low / mid / high** — keeping the dataset
  fixed. Anchor low and high to the cookbook's own values, add one mid point:
  - low:  `--max-concurrency 1`    (cookbook latency point)
  - mid:  `--max-concurrency 32`
  - high: `--max-concurrency 100`  (cookbook throughput point)
- Scale `--num-prompts` so each level runs long enough to be stable
  (e.g. 10 / 300 / 500). Save the full `bench_serving` stdout per level under
  `bench/`.

## 3. Profile

- Capture a torch profiler trace of the serving **forward pass** under a
  representative load (mid concurrency is the default profiling point; profile
  high too if the kernel mix shifts). Use the server profiler
  (`/start_profile` + `/stop_profile`) so the trace reflects real serving, not a
  synthetic single batch.
- Keep raw traces under `profile/` (gzip them; they are large — do not stage
  multi-hundred-MB raw traces for the PR, keep the parsed CSV/MD instead).

## 4. Kernel-workflow inventory (the deliverable)

Run `scripts/extract_kernel_workflow.py` on the trace to produce
`docs/kernel_workflow.md` (+ `.csv`). The inventory ranks GPU kernels by share
of total GPU kernel time.

**Record** every kernel with **≥1%** of GPU kernel time, except:

- **Excluded entirely** (reported only as an aggregate line, never as a task):
  - attention kernels — flash-attn / fmha / mha / mla / paged-attention / `attn`
  - cuDNN kernels — anything `cudnn*`

**Categories** (for the kept kernels):

| Category | Examples | Opportunity |
|---|---|---|
| `gemm` | cutlass/cublas sgemm/hgemm/bf16 gemm, matmul | yes |
| `quant_gemm` | fp8/int8/nvfp4/mxfp4 scaled_mm, w8a8, marlin, blockwise | yes |
| `moe` | fused_moe, grouped/group gemm, expert, topk routing | yes |
| `norm` | rmsnorm, layernorm | yes (often memory-bound) |
| `rope` | rotary embedding | yes |
| `memory_bound` | elementwise, activation (silu/gelu), add/residual, copy/cast, quant/dequant, reduce | yes |
| `comm` | all_reduce / all_gather / reduce_scatter / alltoall / nccl | maybe (overlap, not kernel speed) |
| `other` | anything unclassified ≥1% | review |

A kernel is an **acceleration opportunity** if it is kept (≥1%, not
attention/cuDNN). Each opportunity kernel becomes a task card under
`kernels/<kernel-task>/` in the diffusion task format (prompt / baseline /
solution / bench / docs), seeded with its shapes/dtypes from the trace.

## 5. Cleanup

After the model's folder is committed (inventory + task cards + scripts), the
downloaded model weights are deleted from the remote box. Only delete the
weights/cache for **that** model, and only after the folder is committed and
pushed. Never delete another model's weights or shared caches without asking.
