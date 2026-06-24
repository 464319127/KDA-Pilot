# Benchmark Method — `sgl_kernel.fp8_scaled_mm`

## ABI (both sides identical)
Local direct-symbol TVM-FFI (`TVM_FFI_DLL_EXPORT_TYPED_FUNC` / `tvm::ffi::TensorView`),
destination-passing (output pre-allocated, passed last; functions return void),
every launch on `at::cuda::getCurrentCUDAStream()`. Baseline + candidate compiled
in ONE module (`fp8_scaled_mm_ext`) via `tvm_ffi.cpp.load` (`bench/build_ext.py`).
Baseline = `baseline/fp8_scaled_mm_baseline.cu::fp8_scaled_mm_baseline` (compiles
the verbatim recovered `fp8_gemm_kernel.cu`); candidate =
`solution/fp8_scaled_mm_candidate.cu::fp8_scaled_mm_candidate`.

## Compile flags (symmetric; no one-sided fast-math)
`extra_cflags`: `-std=c++17 -O3`
`extra_cuda_cflags`: `-std=c++17 -O3 --expt-relaxed-constexpr --expt-extended-lambda
-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_VERSIONS_GENERATED -DCUTLASS_TEST_LEVEL=0
--threads=1 -lineinfo -gencode=arch=compute_100a,code=sm_100a`
Include dirs: torch cuda includes + CUTLASS@57e3cfb4 (`include`, `tools/util/include`)
+ `baseline/` (math.hpp, utils.h, cutlass_extensions/) + `bench/csrc/` (abi header).
`-lineinfo` only adds source attribution for ncu (no codegen/numeric effect); applied to both sides.

## Timing
`bench/benchmark.py` = byte-identical copy of `llm/docs/standalone_llm_benchmark_template.py`
(unmodified). Defaults: warmup_runs=10, num_trials=7, inner-loop auto-calibrated to
~1000 µs (min 1, max 4096), isolated spawn subprocess per workload, fresh seeded
inputs per trial, pre-allocated + poisoned outputs, CUDA-event timing, interleaved
A/B order. Per-workload speedup = baseline_median_us / candidate_median_us; headline
= equal-weight geomean over `production:true` rows (call-/time-weighted reported as
secondary). Workloads frozen in `bench/workloads.json` (286 production + 5 edge).

## Tolerances
fp32-dequant oracle; bf16 atol=0.07 rtol=0.02 (covers fp8 accumulation-order
differences across K). Candidate checked vs oracle AND baseline; route==0 fallback
must be bit-identical to baseline.

## GPU / provenance
`ion-b200` GPU id 3 (B200 sm_100), CUDA 13.0 / nvcc V13.0.88, torch 2.11.0+cu130,
tvm_ffi, CUTLASS@57e3cfb47a2d9e0d46eb6335c3dc411498efa198. GPU verified idle
(util 0%, mem 0 MiB) before and after each measured run (see `docs/run_log.md`).
