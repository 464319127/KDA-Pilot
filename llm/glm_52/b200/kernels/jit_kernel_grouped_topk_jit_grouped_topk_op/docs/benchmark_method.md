# Benchmark Method

## Build (identical for baseline and candidate)

Both sides are header-only TVM-FFI modules built with `tvm_ffi.cpp.load_inline`
(the same mechanism upstream `sglang.jit_kernel.load_jit` uses), via
`bench/_jit_build.py`. No live SGLang import; the copied source under
`baseline/` and `solution/` is compiled directly.

- C++ flags: `-std=c++20 -O3`
- CUDA flags (B200 / sm_100): `-DSGL_CUDA_ARCH=1000 -std=c++20 -O3 --expt-relaxed-constexpr`
  with `TVM_FFI_CUDA_ARCH_LIST=10.0`
- Include paths: `baseline/include` (copied `sgl_kernel/*` headers, shared by both
  sides for parity) + the `apache-tvm-ffi` package include dir.
- Exported symbol: `TVM_FFI_DLL_EXPORT_TYPED_FUNC(grouped_topk, (grouped_topk))`
  on both sides. Identical `tvm::ffi::TensorView` signature, destination-passing
  output policy, and `host::LaunchKernel(..., device)` stream resolution
  (`TVMFFIEnvGetStream` = the caller's current CUDA stream).

No one-sided `--use_fast_math`; no asymmetric flags. `fast_sigmoid`, the packed
`(value, index)` comparator, and the renormalization reduction are bit-identical
between the two sides, so candidate output equals baseline output exactly.

## ABI parity

`bench/adapter.py` calls both sides through the identical signature
`grouped_topk(scores, bias, topk_values_out, topk_indices_out,
num_expert_group, topk_group, topk, renormalize, scaling_factor)`. Neither
`call_baseline` nor `call_candidate` allocates outputs (the harness preallocates
and poisons them). Inputs are contiguous fp32 (the op's captured inputs are 100%
`is_contiguous=True`; the upstream Python wrapper calls `.contiguous()` before the
op).

## Timing harness

`bench/benchmark.py` is the project template
(`llm/docs/standalone_llm_benchmark_template.py`) copied verbatim (unmodified).
Timing rules: CUDA events around N back-to-back invocations (inner-loop
amplification), per-trial fresh random inputs, warmup before measurement,
interleaved A/B order per trial, median/mean/std/min/p10/p90 reported,
`speedup = baseline_median / candidate_median`, headline = equal-weight geometric
mean over `production` rows.

### Settings used for the authoritative run

`--target-sample-us 5000 --num-trials 21 --warmup-runs 10 --no-isolated`
(`inner-iterations-max 4096`, default).

**Why `--no-isolated`.** This kernel is sub-microsecond on the GPU and bounded by
the CPU→TVM-FFI→`cudaLaunch` dispatch path (~6.15 µs/call floor on decode). The
template's default per-workload **isolated subprocess** mode injects fresh-process
CPU-launch jitter that contaminates this floor: it produced bimodal, irreproducible
decode ratios (some rows 0.80, others 0.999) for a regime where the candidate runs
the *identical* baseline kernel and must be exactly 1.0. Running all workloads in
one warm process with strong amplification (`target_sample_us=5000`, 21 trials) is
stable and reproducible, and agrees with a controlled direct CUDA-event
micro-benchmark (decode = 1.000, prefill wins). Both sides are measured identically
and interleaved within the same process, so the A/B comparison stays fair. The
isolated mode remains available (drop `--no-isolated`); it is reliable for the
prefill rows (which agree across modes) but not for the decode launch-floor.

## Tolerances

Top-k indices: exact-match (ordered). Weights: fp32 `atol=rtol=1e-5`. Candidate
vs baseline is in practice bit-identical (same kernel math); the independent
oracle cross-check uses the same tolerance. See `bench/correctness.py`.

## GPU

B200, pinned `CUDA_VISIBLE_DEVICES=0` for every correctness, benchmark, and NCU
run. GPU-0 idle verified before/after (see `run_log.md`).
