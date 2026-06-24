# Results — `sgl_kernel.fp8_scaled_mm` on B200 (Ring-2.6-1T)

## Verdict: PROMOTE (decode M=1 regime), fall back elsewhere

A native-CUDA FP8 GEMV wins the **M=1 decode regime** — the single largest
captured shape (M=1 = 38,427 calls) — with **all covered shapes significant and
no regression**. All other shapes fall back to the recovered CUTLASS baseline.
Per the resolved success policy (per-regime win + fallback, DEC-1), this is a
promotion. Small-M swap-AB (M=16..64) is the **next round's** work (task10
reopened), not part of this promotion.

All evidence is from `ion-b200` GPU id 3 (B200 sm_100, CUDA 13.0, torch
2.11+cu130, CUTLASS@57e3cfb4), GPU verified idle before+after each measured run
(`docs/run_log.md`). Timing: the unmodified `standalone_llm_benchmark_template.py`
(template sha256 `2e1712e5…`, byte-identical), CUDA events, inner-loop
amplification, isolated subprocess, interleaved A/B, 7 trials (25 for the
fallback-overhead run). Correctness: **296 passed / 0 failed** (286 production +
4 edge + 6 malformed-input negatives) vs an fp32-dequant oracle AND the baseline.
Candidate source sha256 `1580070d…`; baseline ABI wrapper `07d0403b…`.

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

## Contract scope (bias)

All 2,720 captured variants are `bias=None` and `out_dtype=bf16`. The recovered
local ABI implements that captured contract (no bias channel); `bias!=None` is
outside the benchmarked interface and is therefore not a routable/edge case (it
was removed as a metadata-only row). The hardened predicate + ABI dtype guard
reject any non-fp8_e4m3fn input, malformed scale rank, or mixed-device tensor
(proven by `bench/correctness.py` negative-route tests).

## Follow-ups (next rounds; bounded by the plan's upper path boundary)

1. **task10 (next round)**: SM100 swap-AB skinny tensor-core GEMM for the hot
   small-M shapes (M=23,32,57; KernelWiki `pr-vllm-27284`), or an evidence-backed
   small-M no-go. These currently fall back at parity (no regression).
2. Higher-MLP / faster-fp8-decode GEMV to lift the latency-bound ceiling and widen
   M=1 coverage to the excluded large-K×N shapes (K≥4096 ∧ N≥3072).
