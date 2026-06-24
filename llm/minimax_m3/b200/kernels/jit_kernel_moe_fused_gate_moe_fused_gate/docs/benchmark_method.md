# Benchmark Method — `moe_fused_gate`

## Build (both sides identical)
- Mechanism: TVM-FFI `tvm_ffi.cpp.load_inline` (`bench/_jit_build.py`) — the same header-only
  path SGLang's `jit_kernel` uses. Baseline `#include`s `baseline/csrc/moe/moe_fused_gate.cuh`;
  candidate `#include`s `solution/csrc/moe/moe_fused_gate_candidate.cuh`. Each exports the
  identical typed symbol `TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_fused_gate, (MoEFusedGateKernel::run))`.
- Shared include path: `baseline/include` (the copied `sgl_kernel` headers), for both sides.
- Compile flags (symmetric, no one-sided fast-math): cflags `-std=c++20 -O3`; cuda
  `-DSGL_CUDA_ARCH=1000 -std=c++20 -O3 --expt-relaxed-constexpr`; `TVM_FFI_CUDA_ARCH_LIST=10.0`
  (B200 sm_100). No architecture/math flags applied to only one side.

## ABI parity
Both sides take the identical signature `run(input, bias, output, indices, topk, scoring_func,
num_fused_shared_experts, renormalize, routed_scaling_factor, apply_routed_scaling_factor_on_output)`
with `TensorView` inputs first and caller-preallocated `output`/`indices` last
(destination-passing). Neither call allocates output tensors. Both launch via `host::LaunchKernel`
(PyTorch current stream). `bench/adapter.py` resolves candidate availability once at import (no
asymmetric per-call filesystem stat that would bias launch-bound timing).

## Timing harness
- `bench/benchmark.py` is a **verbatim copy** of `llm/docs/standalone_llm_benchmark_template.py`
  (unmodified). CUDA-event GPU timing; inner-loop amplification to a target sample of ≥~1000 µs
  (critical for these sub-µs launch-bound kernels); interleaved A/B per trial; warmup before
  measurement; fresh per-trial random inputs; the timed region excludes input generation, JIT
  build, imports, and allocation. Reports median/mean/std/min/p10/p90 per workload; headline =
  equal-weight geometric mean of per-workload speedup over `production:true` rows.
- Authoritative prefill run: `python benchmark.py --only <11 prefill ids> --num-trials 9 --no-isolated`.
  - `--no-isolated` justification: the kernel is at the CPU-launch floor; isolated per-workload
    subprocesses inject irreproducible scheduling/IO jitter that swamps the ~6–8 µs signal. One
    process gives stable interleaved A/B samples. (Same rationale as the sibling grouped_topk task.)
- Context priming: `bench/adapter.py::_prime_context` runs ONE large-path baseline launch at import
  (before any timed region) to establish a warm CUDA context. It deliberately does NOT launch the
  baseline decode path (that path is UB — see below).

## Tolerance policy
- Selected expert **indices: exact-match** (ordered) vs the oracle/baseline.
- Gathered **weights: `atol=rtol=1e-5`** (fp32 contract default). Weights use the kernel's exact
  fp32 op order (`weight/norm*scale`; shared-slot weight `(routed_sum/rsf)/norm*scale`, not
  hardcoded to 1.0).

## Decode-baseline handling (why the headline is prefill-scoped)
The recovered baseline **decode** (small-token, `num_experts=128`) kernel reads uninitialized
shared memory and faults nondeterministically (`CUDA illegal memory access`) — see
`docs/baseline_source.md` and `docs/results.md`. It is therefore **not reliably benchmarkable**.
Consequently:
- Baseline-vs-candidate speed ratios are reported on the **prefill** rows only (baseline reliable).
- **Decode correctness** is validated candidate-vs-**oracle** (the ground-truth reference), not
  candidate-vs-baseline, in `bench/correctness.py`.
- **Decode candidate latency** is measured by the committed candidate-only script
  `CUDA_VISIBLE_DEVICES=4 python bench/bench_decode_candidate.py` (same CUDA-event + inner-loop
  amplification + median-over-trials methodology as the template), reported candidate-absolute in
  `docs/results.md`. No speed ratio is fabricated for a path the baseline cannot run.
This is the only deviation from a uniform candidate-vs-baseline comparison, and it is forced by a
genuine baseline bug rather than a methodology choice.

## Input contract
All 296 captured variants are finite float32 (~randn-scale). **NaN/Inf inputs are out of contract**:
the baseline ignores NaN in its `>` comparisons while the candidate's packed-key comparison may
order NaN differently, so the two are not matched on NaN/Inf. Correctness is defined on finite
inputs (validated by `bench/correctness.py`, including a subnormal-stress row).

## Aggregates reported (see docs/results.md)
- **Prefill equal-weight geomean** of per-row `baseline_median/candidate_median` over the 11
  prefill production rows; and **call-count-weighted geomean** (`exp(Σ cᵢ·ln sᵢ / Σ cᵢ)`). For this
  task all prefill rows share `call_count=456`, so the two coincide (1.0105).
- **Decode speedup = N/A** and **production-wide (decode+prefill) speedup = N/A**: the baseline
  decode path is UB/unbenchmarkable, so any decode-containing ratio has an invalid denominator.
  Decode is reported as candidate-absolute latency instead. (Decode is 84.2% of production calls,
  so the absence of a production-wide ratio is a genuine limitation, stated explicitly — not a
  silent omission.)

## Profiling
NCU `--set basic` on the candidate (GPU 4): SM compute ~0.06 %, DRAM ~0.02 %, memory ~3 %,
achieved occupancy ~12.5 % → launch/latency-bound. Warp-specialization-report-skill not applicable
(no producer/consumer pipeline). See `docs/results.md`.
