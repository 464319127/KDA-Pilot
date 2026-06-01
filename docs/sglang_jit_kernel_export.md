# SGLang jit_kernel Export Contract

KernelPilot diffusion tasks should now promote kept CUDA kernels into SGLang's
native `jit_kernel` stack.

## When to export

Export is the **final step, run only after the RLCR loop finishes** —
correctness passing on every configured shape and benchmark evidence
recorded. It is not part of the optimization rounds. After exporting, verify
the kernel **drop-in replaces** the corresponding kernel inside an editable
SGLang checkout: the public entry point resolves to the candidate, the task's
correctness oracle passes inside SGLang, a smoke benchmark shows
parity-or-speedup vs the original SGLang kernel, and unsupported signatures
still fall back to the SGLang baseline.

## Required Integration Shape

- Put promoted device code in `python/sglang/jit_kernel/csrc/.../*.cuh`.
- Expose it through `python/sglang/jit_kernel/...` Python wrappers that use
  `load_jit`, `cache_once`, and `make_cpp_args`.
- Export a templated `...Kernel<...>::run` launcher that accepts
  `tvm::ffi::TensorView` arguments.
- Validate tensors with `TensorMatcher` / symbolic sizes, dtypes, strides, and
  CUDA device checks before launch.
- Launch on the current stream with `host::LaunchKernel(...).enable_pdl(...)`
  when PDL is part of the template configuration.
- Preserve the existing public SGLang callable names. Unsupported signatures
  must fall back to the original SGLang baseline.

## Compile flags and PDL

- **Match the corresponding SGLang `jit_kernel` kernel's compile options.** In
  particular, do **NOT** pass `--use_fast_math`: SGLang's `jit_kernel` build
  does not use it, so adding it diverges from the baseline's numerics and is
  not a fair/consistent comparison. Add an extra `nvcc` flag only if the
  matching SGLang kernel also uses it.
- **PDL may be tried** (the baseline templates `enable_pdl` via
  `is_arch_support_pdl()`), but it is **optional** and must be validated on the
  real workload. In the qknorm pilot, enabling PDL *hurt* isolated-launch
  latency, so do not assume it helps — keep it only if it wins on the task's
  actual benchmark.

## Benchmark Rule

Benchmark evidence must compare steady-state baseline and candidate calls:

- run one-time JIT compile, SGLang JIT build, autotune, imports, cache
  population, and callable binding before timed samples;
- run warmup calls for both baseline and candidate before repeated timing;
- record warmup count, iteration count, exact command, GPU id/model, host,
  candidate source version, and SGLang commit;
- exclude input restore/copy and Python setup from the timed region;
- report median, mean, std, min, p10, p90, per-shape speedup, and all-shape
  geometric mean speedup.

## Per-Task Evidence

Before ending an RLCR turn, write `docs/sglang_jit_export.md` inside the task
folder with:

- SGLang files to patch;
- public entry points preserved;
- template arguments and CUDA wrapper names passed to `load_jit`;
- arch/shape/dtype gates and fallback behavior;
- correctness tests and benchmark commands used for validation.
