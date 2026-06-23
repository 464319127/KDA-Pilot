# Results — fast_topk_transform_fused (B200) — measured: NO-GO (candidate ≈ baseline)

> **FINAL STATUS — NO FURTHER OPTIMIZATION ACHIEVABLE (on these captured shapes).** This kernel was
> benchmarked candidate-vs-baseline on a strictly-idle B200 and shows **no measurable speedup**
> (production geomean ≈ 1.0, within the tiny-kernel noise floor). The dominant decode path is already at
> a fixed ~8 µs store/launch-overhead floor that a faster per-call copy cannot beat; a real win would
> need a different class of change (launch elimination via CUDA Graphs / PDL), which is out of scope
> here. The native candidate is **retained as correctness-clean and dispatch-robust but NOT promoted as
> a speedup.** Marked closed: no better optimization expected for this kernel/shape set.

> **Headline:** the native CUDA candidate is correctness-clean (`matched_ratio==1.0`, 251/251) but
> delivers **no measurable speedup** over the recovered baseline. On the production grid the
> candidate-vs-baseline **geomean speedup is ≈ 1.0** (0.9969 in the clean full-grid run; 1.0014 in an
> earlier 235-row run) with a per-row spread of **0.82–1.15** — i.e. the difference is entirely within
> the tiny-kernel measurement noise floor, and roughly half the rows land above 1.0 and half below
> (98/236 > 1.0). **Decision: NO-GO** — the optimization does not beat the baseline.

## Provenance (AC-7 note: GPU deviation, user-authorized)
- Measured on **B200 GPU id 2** (host `ion-b200`, container `sglang_bbuf`), **strictly idle**: `0%`
  util and `0–4 MiB` used, **both before and after** the run, no compute processes (verified via
  `nvidia-smi` summary + compute-apps). Recorded in `docs/run_log.md`.
- The original plan pinned timing to **GPU id 1** (AC-7), which carried a persistent ~59.6 GB parked
  tenant process (util 0%) across the whole RLCR loop and never reached strict idle. On **2026-06-24
  the user explicitly authorized using any idle GPU** to obtain the measurement; GPU 2 was the
  strictly-idle card. This is a documented, user-authorized deviation from the GPU-1 pin — not a
  fabrication; the exact GPU used is recorded here and in the run log.
- Toolchain: torch `2.11.0+cu130`, CUDA `13.0`, `tvm_ffi 0.1.9`, sm_100. Harness:
  `bench/benchmark.py` (template-identical), CUDA-event timing, interleaved A/B, isolated subprocess
  per workload, warmup=10 / trials=7, inner-loop amplification ~1000 µs.

## Measured speedups (clean run: 251/251 PASSED, 236 production rows timed)

| Group | n | geomean speedup | min | max |
|---|---|---|---|---|
| **Production (all)** | 236 | **0.9969** | 0.8234 | 1.1492 |
| — native bucket (decode copy/fill) | 212 | 0.9963 | 0.8234 | 1.1195 |
| — baseline fallback (radix / large-prefill) | 24 | 1.0028 | 0.9820 | 1.1492 |
| Regression grid | 15 | 0.9964 | 0.9732 | 1.0164 |

> speedup = baseline_median / candidate_median (>1.0 = candidate faster). Production geomean is the
> AC-8 headline. A second earlier run (235 rows; one large radix row was excluded by a since-fixed
> sanity-gate limitation) gave geomean 1.0014 — the ~0.45% run-to-run difference is itself within the
> noise, reinforcing "no systematic effect."

### Per-regime median latency (production)
| Regime | n | baseline p50 | candidate p50 |
|---|---|---|---|
| decode (naive, `S==B`, dominant) | 212 | 8.17 µs | 8.23 µs |
| radix (`length>topk`, large-B prefill) | 24 | 9.64 µs | 9.61 µs |

## Why no speedup (active-bound reasoning)
- **Decode naive (the bucket the candidate specializes, ~8 µs):** this path is a pure memory-bound
  copy/fill — write all 2048 int32 outputs/row (8 KB) and gather `src_page_table[b, :length]`. The
  recovered baseline already moves exactly those bytes. The candidate's rewrite (coalesced int32 stores
  + multiple CTAs per row for occupancy) does **not reduce the bytes moved or the fixed kernel-launch
  overhead**, so both implementations sit at the same ~8 µs store/launch-overhead floor and the
  candidate-vs-baseline delta is pure noise. This is consistent with the `docs/dispatch.md` hypothesis
  that the regime is store/launch-bound; it is **inferred from the flat latency + memory-bound nature**,
  not from an NCU profile (NCU was not run — see Limitations).
- **Radix / large-prefill (24 rows):** the candidate has no native radix path and **falls back to the
  baseline**, so its speedup is ≈ 1.0 by construction (geomean 1.0028, within noise).

## Decision: NO-GO (evidence-backed)
Per AC-8, a *win* requires production geomean > 1.0 with full correctness. The measured geomean is
≈ 1.0 (0.9969), below the win bar and inside the noise floor, so this is a **no-go**, supported by:
baseline numbers (above) + a reasoned native attempt (the decode copy/fill bucket) + full correctness
(251/251) + benchmark evidence (two strict-idle runs) + a named bound (store/launch-overhead-bound
decode). The candidate is retained as correctness-clean and dispatch-robust (it never regresses: it
either matches the baseline within noise or falls back to it), but it is not promoted as a speedup.

## Limitations / what would change the verdict
- **No NCU profile** was collected; the active-bound is inferred, not counter-measured. An NCU run on
  the decode path would confirm/refute the store/launch-overhead diagnosis.
- The decode path is already at a ~8 µs floor dominated by fixed per-launch overhead for tiny batches;
  a genuine win would likely require **eliminating launches** (CUDA Graphs / PDL across the many tiny
  decode calls) rather than making the per-call copy faster — a different optimization than this bucket.
- Non-contiguous `score` strides are surrogate (the capture records only contiguity); the candidate
  does not read `score` on the decode path, so this does not affect the decode result.
