# Benchmark Method — `sgl_kernel.topk_sigmoid`

## Kernel under test

MoE sigmoid-gated top-8-of-288 router. Recovered baseline = upstream SGLang `main`
@ `5e6d7c1615a95dc5f98e69b4b18af0ae160b10b8` (`sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.cu`,
vendored verbatim). Candidate = single fused native-CUDA kernel
(`solution/topk_sigmoid_candidate.cuh`). See `docs/baseline_source.md` for the exact semantics.

## Local ABI (both sides identical)

Local direct-symbol TVM-FFI: `TVM_FFI_DLL_EXPORT_TYPED_FUNC`, `tvm::ffi::TensorView` arguments +
`int64_t` scalars, destination-passing (`topk_weights`/`topk_indices` written in place,
`gating_output` read-only, returns void), every launch on `at::cuda::getCurrentCUDAStream()`.
Baseline (`topk_sigmoid_baseline`) and candidate (`topk_sigmoid_candidate`) are exported from ONE
module compiled together; both receive the identical 5-argument TensorView signature and go
through the same `bench/adapter.py` call path, so wrapper/marshalling overhead is symmetric. The
baseline export bridges TensorView → `torch::Tensor` (`torch::from_blob`, no copy) and calls the
vendored `topk_sigmoid(...)`; no live SGLang is imported at runtime.

## Compile flags (symmetric; recorded for provenance)

- `extra_cflags`: `-std=c++17 -O3`
- `extra_cuda_cflags`: `-std=c++17 -O3 -gencode=arch=compute_100,code=sm_100`
- `extra_ldflags`: torch lib paths + rpath + `-lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda`
- **No `--use_fast_math`** on either side (no one-sided numeric/codegen flags).
- Build: `tvm_ffi.cpp.load` into `bench/.build/` (JIT, outside any timed region).

## Timing rules (from `bench/benchmark.py`, the copied standalone template)

- CUDA events for GPU time; per-call wall-clock recorded only as a secondary diagnostic.
- Inner-loop amplification: N back-to-back invocations inside one event pair, divided by N;
  N doubled from 1 up to 4096 until a sample reaches `target_sample_us = 1000`.
- `num_trials = 7`, `warmup_runs = 10` discarded per trial; interleaved A/B order per trial.
- Each workload runs in an isolated `spawn` subprocess. Outputs poisoned (NaN/-17) before each
  call; correctness checked before timing (a row that fails correctness is not timed).
- Per side: median/mean/std/min/p10/p90 µs. Per-workload speedup = `baseline_median / candidate_median`.
- **Headline = equal-weight geometric mean of speedup over `production: true` workloads.**

## Tolerances

fp32 weights: `atol = rtol = 1e-5`. Selected expert ids: **exact match** (integer/top-k contract).
`bench/adapter.py::compare_outputs` enforces exact int32 id equality + fp32 weight tolerance (the
template's default element-wise comparator would wrongly apply atol/rtol to ids).

## Workloads

`bench/workloads.json` is frozen, deduplicated from `docs/evidence.json` by `bench/gen_workloads.py`:
24 production rows (one per distinct captured token count N ∈ {1,7,12,15,16,18,19,24,27,32,34,44,53,
55,61,70,72,80,1579,9030,10207,16206,16474,16883}, fp32/contiguous/topk=8/experts=288/renormalize=True/
bias present, carrying the aggregate captured call count) + 6 regression/fallback rows
(non-contiguous, fp16, bf16, experts=64, topk=4, renormalize=False) that must route to the baseline.

## Promotion bar (DEC-1) and decode timing (DEC-2)

- DEC-1: promotion judged on the equal-weight geometric-mean op-latency speedup over production
  workloads at the strict op ABI; call-weighted aggregate saving is secondary context only.
- DEC-2: per-call CUDA-event timing with inner-loop amplification is the sole basis; no
  CUDA-graph-replay view.
- DEC-3: correctness uses synthetic seeded `correction_bias` (the recovered baseline is the oracle;
  real checkpoint values are not required for baseline-equivalence).
