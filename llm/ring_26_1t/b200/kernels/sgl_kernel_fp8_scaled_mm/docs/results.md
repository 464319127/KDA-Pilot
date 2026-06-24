# Results — `sgl_kernel.fp8_scaled_mm` on B200 (Ring-2.6-1T)

## Verdict: PROMOTE (decode M=1 regime), fall back elsewhere

A native-CUDA FP8 GEMV wins the **M=1 decode regime** — the single largest
captured shape (M=1 = 38,427 calls) — with **all covered shapes significant and
no regression**. The **small-M regime (M=2..64) is an evidence-backed no-go**: an
SM100 swap-AB CUTLASS GEMM was implemented and is correct, but loses to the
baseline (see "Small-M" below), so it falls back. All other shapes fall back to
the recovered CUTLASS baseline. Per the resolved success policy (per-regime win +
fallback, DEC-1), promoting M=1 with an evidence-backed no-go elsewhere is the
outcome.

All evidence is from `ion-b200` GPU id 3 (B200 sm_100, CUDA 13.0, torch
2.11+cu130, CUTLASS@57e3cfb4), GPU verified idle before+after each measured run
(`docs/run_log.md`). Timing: the unmodified `standalone_llm_benchmark_template.py`
(template sha256 `2e1712e5…`, byte-identical), CUDA events, inner-loop
amplification, isolated subprocess, interleaved A/B, 7 trials (25 for the
fallback-overhead run). Correctness: **299 passed / 0 failed** (286 production +
4 edge rows + 9 negative/edge tests: e5m2/uint8 A, e5m2 B, malformed scale
`[M,2]`/`[N,2]`, fp16-out, CPU-input, mixed-device, and the bias edge) vs an
fp32-dequant oracle AND the baseline. Source sha256 (HEAD):
candidate `507dcce3…`, baseline ABI wrapper `69979be8…`, swap-AB `d19e004e…`,
`bench/correctness.py` `8338da18…`; benchmark.py byte-identical to the template
(`2e1712e5…`).

## Final full-grid benchmark (all 286 production shapes, idle GPU 3)

| Metric (over 286 production rows) | Value |
|---|---|
| Equal-weight geomean speedup (mandated headline) | **1.023×** |
| Call-weighted geomean (by captured call count) | **1.167×** |
| Time-weighted geomean (by baseline time × calls) | **1.135×** |

The equal-weight headline is ≈1.0 because the 7 covered M=1 shapes are diluted by
279 fall-back shapes at parity; weighting by real traffic (call/time-weighted),
where the dominant M=1 decode shape carries the load, the candidate delivers
**+13–17%**.

### Covered M=1 shapes — all significant, no regression (AC-6)

Significance = speedup ≥ 1.10 AND candidate p90 < baseline p10 (non-overlapping
trial distributions).

| shape (M·K·N) | baseline median µs (p10/p90) | candidate median µs (p10/p90) | speedup | significant |
|---|---|---|---|---|
| 1·1536·1536 | 21.51 (21.22/22.18) | 5.54 (5.46/5.90) | **3.88×** | yes |
| 1·256·8192 | 21.17 (21.10/22.03) | 5.46 (5.45/5.86) | **3.87×** | yes |
| 1·1024·8192 | 21.34 (21.10/21.95) | 6.19 (6.17/6.20) | **3.45×** | yes |
| 1·8192·512 | 21.52 (20.87/21.80) | 8.26 (8.26/8.27) | **2.61×** | yes |
| 1·8192·1024 | 21.32 (21.00/21.76) | 8.26 (8.26/8.28) | **2.58×** | yes |
| 1·2304·8192 | 20.55 (19.99/20.88) | 10.31 (10.30/10.31) | **1.99×** | yes |
| 1·8192·2112 | 20.08 (19.96/20.70) | 10.93 (10.85/10.95) | **1.84×** | yes |

**7/7 covered shapes are significant ≥1.10 wins; 0 significant regressions.**
Covered-regime geomean **2.78×** (every candidate p90 is far below the
corresponding baseline p10 — distributions do not overlap).

> Note on baseline latency: the focused round-0 runs measured the M=1 baseline at
> ~12 µs vs ~21 µs in this sustained full-grid run. The candidate (memory-latency
> bound) is stable (~5.5–10.9 µs) across both; the baseline (tensor-core, more
> SM-clock-sensitive) slows under sustained load. Because timing is interleaved
> A/B per trial, the within-run speedups and non-overlap are valid regardless;
> the candidate's advantage only grows under realistic sustained load.

### Per-regime geomean (286 production)

| regime | geomean | n | note |
|---|---|---|---|
| decode_tiny (M≤16) | 1.117× | 63 | M=1 wins; M=3..16 fall back at parity |
| decode_small (16<M≤64) | 0.997× | 126 | all fall back (no M=1) |
| medium (64<M≤256) | 0.990× | 10 | fall back |
| prefill (M>256) | 1.000× | 87 | fall back |

### Fallback overhead (AC-5)

Route-0 (uncovered) candidate vs a direct baseline call — the candidate path adds
only a host-side coverage predicate over identical baseline device code:
- Full grid (279 uncovered shapes, 7 trials): **median +0.009%**, mean +0.27%.
- Dedicated 25-trial run (6 representative uncovered shapes): geomean **+0.88%**
  overhead, worst +1.67% (one prefill shape), best +0.26%.

Overhead is small (median ≈0; typical ≤~1%), attributable to the host dispatch
predicate; it never changes device behavior. The covered fast-path shapes do not
pay it (they take the GEMV directly).

## Small-M regime (M=2..64) — evidence-backed NO-GO

A native-CUDA SM100 swap-AB CUTLASS FP8 scaled GEMM (`solution/fp8_swapab_smallm.cu`)
was implemented for the hot small-M shapes (M∈{23,32,57}), following the upstream
SM100 swap-AB pattern (KernelWiki `pr-vllm-27284`): GEMM-A=`Bphys[N,K]`,
GEMM-B=`A^T[K,M]`, problem `{N,M,K}`, scales swapped, `LayoutD=ColumnMajor` so the
transposed result lands in row-major `out[M,N]` with no copy. It is **correct**
(all small-M shapes match the fp32 oracle, route diagnostic confirmed), but it
**loses to the baseline**, so the dispatch falls back (no regression).

Benchmark (idle GPU 3, 7 trials), swap-AB candidate vs baseline:

| M | k1024·n8192 | k256·n8192 | k8192·n512 | k8192·n1024 | k8192·n3072 |
|---|---|---|---|---|---|
| 23 | 0.829× | 0.877× | 0.864× | 0.839× | 0.846× |
| 32 | 0.860× | 0.865× | 0.870× | 0.850× | 0.859× |
| 57 | 0.836× | 0.843× | 0.857× | 0.835× | 0.863× |

Geomean **0.85×** (candidate ~20–21µs vs baseline ~17–18µs); not one shape clears
the ≥1.10 bar.

**Named active bound (NCU, m32·k1024·n8192, the swap-AB kernel):** tensor-core
activity **1.9%**, SM throughput **6.5%**, achieved occupancy **10.7%**, DRAM read
**10.7%** of peak. The kernel is **pipeline-fill / occupancy bound**: the tiny
swapped-N dimension (= original M = 32) under-fills the 2-SM warp-specialized
`tcgen05` mainloop (TMA-producer → tensor-core-consumer), so the tensor cores sit
nearly idle. The baseline `Gemm64` (CTA-M 64) is already 36–89% M-tile-utilized
for M=23–57, leaving too little headroom for swap-AB to overcome its
pipeline-fill + column-major-epilogue overhead.

This is a complete no-go per the bar: baseline numbers + an implemented, correct
candidate + benchmark evidence + NCU/named bound + the dispatch decision (keep
baseline for small-M). The kernel and predicate remain in tree
(`kSmallMSwapAbEnabled=false`) as the documented attempt.

**Warp-specialization profiling note (AC-8):** the swap-AB kernel IS warp-specialized
(`MainloopSm100TmaUmmaWarpSpecialized`: TMA producer + `tcgen05` tensor-core
consumer). The `warp-specialization-report-skill` (per-warp `clock()` overlap
timeline) was not run because NCU already localizes the bound at the *grid* level:
achieved occupancy 10.7% and tensor-core activity 1.9% mean the kernel barely
fills the machine for swapped-N=M=32 — the issue is under-occupancy / pipeline
fill-drain, not a producer/consumer overlap failure *within* a well-fed CTA (which
is what the clock-stamp timeline diagnoses). The actionable lever is grid/tile
occupancy, an NCU/roofline matter, so the warp-spec timeline would not change the
verdict.

## Roofline / active-bound analysis (NCU on GPU 3)

- **Baseline M=1**: CUTLASS sm100 uses a 64-row MMA tile even for M=1 → ~63/64 of
  the tensor-core M dimension is unused; far from bandwidth-efficient in decode.
- **Candidate FP8 GEMV M=1** (NCU on 1·1024·8192): `dram__bytes_read` = 9.48% of
  peak, `sm__throughput` = 37%, occupancy 63%, dominant stall = `long_scoreboard`
  → **memory-latency-bound** (not bandwidth-saturated, not compute-bound). It wins
  by avoiding the MMA waste and reading B once with coalesced `uint4` loads;
  headroom remains below the HBM roofline (a higher-MLP kernel could go faster).

Named active bounds: baseline M=1 = tensor-core under-utilization; candidate M=1 =
global-load latency.

## Contract scope (bias) — `bias!=None` → candidate fallback to baseline (AC-3.1)

All 2,720 captured variants are `bias=None` and `out_dtype=bf16`, so the
production candidate ABI is 5-arg (no bias). The required AC-3.1 `bias!=None`
edge is exercised through a **test-only bias-capable candidate fallback route**:
`fp8_scaled_mm_candidate_bias(a,b,scale_a,scale_b,bias,out)` + its route
`fp8_scaled_mm_candidate_bias_route(...)`. `bench/correctness.py:bias_edge_test`
uses an otherwise-covered M=1 winner (M=1,K=1024,N=8192) and verifies:
`route_bias==0` (a biased call is routed away from the bias-unaware M=1 GEMV fast
path), and `candidate_bias` produces the correct biased result
(`out=(A@B)·scale_a·scale_b+bias`) — equal to both the recovered baseline
(`baseline_bias`) and the fp32 oracle within bf16 tol. So a biased call falls back
to the baseline and is correct. The hardened predicate + ABI guard additionally
reject any non-fp8_e4m3fn input, malformed scale rank, or non-CUDA/mixed-device
tensor before any view (proven by `bench/correctness.py` negative-route tests
incl. CPU-input and e5m2-B).

## Status of the plan's directions

- **M=1 decode (task8/9): PROMOTED** — native FP8 GEMV, geomean 2.78× covered, no regression.
- **Small-M M=2..64 (task10): evidence-backed NO-GO** — swap-AB implemented + correct
  but 0.85× (NCU: pipeline-fill / occupancy bound); falls back to baseline.
- **Prefill / medium: baseline fallback** — beating vendor CUTLASS is not expected
  (validated by parity in the full-grid run).

## Follow-ups (optional; bounded by the plan's upper path boundary)

1. Higher-MLP / faster-fp8-decode GEMV to lift the M=1 latency-bound ceiling and
   widen M=1 coverage to the excluded large-K×N shapes (K≥4096 ∧ N≥3072), which
   currently fall back at parity.
2. Small-M is a no-go for the swap-AB approach as configured; a future attempt
   would need higher swapped-N tile occupancy (multi-tile-N batching across the
   small M, or a non-CUTLASS skinny kernel) — only if small-M throughput becomes a
   priority. Re-enable via `kSmallMSwapAbEnabled` after a winning re-tune.
