# B200 int8_scaled_mm KernelPilot Prompt

Copy this as one end-to-end prompt.

```text
/humanize:humanize-kernel-agent-loop

Use the ion-b200 remote GPU environment for all B200 work. All CUDA, Python,
pip, nvcc, build, test, benchmark, and Nsight Compute commands must run inside
the existing sglang_bbuf Docker container on ion-b200, with GPU0 selected.

Use this command pattern for remote execution:

ssh ion-b200 'docker exec sglang_bbuf bash -lc "CUDA_VISIBLE_DEVICES=0 <command>"'

Do not run Python, pip, nvcc, builds, tests, benchmarks, or profiling directly
on the ion-b200 host.

Task:
Optimize SGLang's int8_scaled_mm kernel on NVIDIA B200 for this focused case:

- M=64
- N=2048
- K=2048
- out_dtype=fp16
- bias=true

Target:
Beat the current SGLang implementation by at least 2.5x on median latency for
the exact same shape, dtype, layout, bias behavior, and B200 GPU0 environment.

Scope:
- Work in the current standalone workspace root. Do not create a nested repo
  unless the current directory is not writable.
- Build a benchmarkable and profileable CUDA/C++ or CUDA inline-PTX candidate.
- Keep the optimization focused on this single shape first.
- Do not change SGLang behavior or public APIs unless a minimal local harness
  requires it for baseline measurement.

Baseline:
- Inspect the current SGLang implementation path for int8_scaled_mm.
- Build a reproducible baseline harness before optimizing.
- Report SGLang baseline latency for the exact focused case.
- Optional secondary baselines are allowed, such as torch._int_mm or fp16 GEMM,
  but the acceptance target is relative to SGLang.

Correctness:
- Compare against the current SGLang result and a PyTorch reference when
  practical.
- Include bias in the validation path.
- Report max absolute error, relative error, and the tolerance used.
- The final candidate must pass correctness before benchmark claims count.

Benchmarking:
- Use warmup and repeated timing.
- Report median latency, mean latency, std, min, p10, p90, and speedup over
  the SGLang baseline.
- Keep benchmark scripts and raw result logs in the workspace.
- Every claimed improvement must identify the candidate commit/file version and
  the command used to produce the result.

Optimization guidance:
- Use KernelWiki when prior B200, SM100, CUTLASS, SGLang, or int8 GEMM evidence
  is useful.
- Use Nsight Compute evidence when a candidate is correct but not clearly
  target-complete.
- Consider B200/SM100-specific paths such as tcgen05 INT8 MMA, TMEM/TMA where
  appropriate, warp specialization, persistent scheduling, Stream-K or split-K,
  cluster shape choices, vectorized loads/stores, shared-memory staging, and a
  fused bias/output epilogue.
- Prefer evidence-backed edits over broad rewrites. Keep a performance map of
  tested variants and rejected ideas.

Completion:
- Continue iterating until the final correct candidate is at least 2.5x faster
  than the SGLang baseline on the focused case, or until at least six substantial
  evidence-backed attempts show why the target is blocked.
- The final report must include baseline numbers, final numbers, speedup,
  correctness tolerances, build/test/benchmark commands, key design decisions,
  and the next most promising follow-up if the target is not reached.
```
