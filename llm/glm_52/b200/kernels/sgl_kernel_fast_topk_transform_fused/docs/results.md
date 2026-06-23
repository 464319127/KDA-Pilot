# Results — fast_topk_transform_fused (B200) — STATUS: candidate ready; win/no-go PENDING strict-idle timing

> **Headline:** the native CUDA candidate is implemented, correctness-verified (`matched_ratio==1.0`,
> 251/251), hardened, and dispatch-robust, but its **measured speedup is not yet determined**. A
> candidate-vs-baseline benchmark and an NCU active-bound profile are **timing-gated** on a strictly-idle
> GPU 1, which has carried a persistent ~59.6 GB parked tenant process (util 0%) across Rounds 5–12. Per
> the user's Round-7 decision to wait for strict idle (rather than time on the parked card), and because
> AC-7 pins all timing to GPU id 1, no candidate timing has been collected. **This is therefore neither a
> win nor a documented no-go yet — it is an explicit PENDING state.** No performance number in this doc
> is fabricated; every figure below is either a provisional baseline measurement or a labeled hypothesis.

## What is established (real evidence)

### Correctness (authoritative gate — AC-4)
- `CUDA_VISIBLE_DEVICES=1 python3 bench/correctness.py cuda:0` → **`matched_ratio = 1.0000 (251/251)`**
  on B200 GPU 1 (236 production + 15 regression rows). Naive rows are exact candidate==baseline==oracle;
  radix rows are valid-top-k (order/tie tolerant via actual-page-table inversion + row-start windows).
- Dispatch split (env-gated diagnostic): **221 calls run the native kernel, 30 fall back** to the
  recovered baseline (the `min(N,M)>2048` radix / large-prefill rows). `solution/_probe.py` adds a
  `fallback_regression` confirming an out-of-contract input (score-batch mismatch) falls back (raises
  like the baseline) rather than silently running the native path.

### Candidate (AC-6)
- `solution/candidate_topk_transform.cu`: native `decode_copy_fill_kernel` for the dominant decode-naive
  bucket (`topk==2048 && row_starts==None && S==B && min(N,M)<=2048`), writing
  `dst[b,col]=(col<lengths[b])?src_page_table[b,col]:-1` over all 2048 columns; one CTA per
  (row, 256-col tile), one thread/column (coalesced int32 stores). Metadata-only dispatch (no host
  `lengths` read, no sync, no hot-path alloc), a non-synchronizing launch-error check, and a dispatch
  predicate that is a **superset of the baseline's metadata contract** (every uncovered shape/param
  falls back). See `docs/dispatch.md`.

### Provisional baseline timing (NOT AC-7-complete; baseline-vs-STUB)
> These numbers were taken in Round 6 on GPU 1 at util 0% **but with the 59 GB parked allocation
> resident** (fails AC-7's strict-idle bar), and the candidate at that time was a **baseline-forwarding
> stub** — so the "speedup" below is baseline-vs-itself (a harness-fairness check), **not** a result for
> the current native candidate. See `docs/benchmark_method.md`.
- Per-regime baseline `median_us` (production): decode (naive, `S==B`) n=212, **p50 9.32 µs** (7.63–13.59);
  radix (`length>topk`, large-B prefill) n=24, **p50 9.29 µs** (8.22–**74.79**).
- Production headline (236 rows, baseline-vs-stub): geomean 0.9977, min 0.80, max 1.097 — i.e. the
  harness is fair and the per-row tiny-kernel noise floor is ~±20%; **a real candidate win must robustly
  clear that floor.**

## Active-bound analysis (HYPOTHESES — not yet confirmed by NCU)
From `docs/dispatch.md` (KernelWiki `pattern-memory-bound` + the captured shapes), to be confirmed/refuted
by NCU once timing is unblocked:
- Decode naive (dominant, ~9 µs): expected **store / launch-overhead bound** — small `(B,2048)` int32
  writes (8 KB/row), tiny batch ⇒ low waves-per-SM. The candidate attacks this via coalesced full-row
  stores + multiple CTAs/row for occupancy; whether that beats the baseline's single 1024-thread row
  block is exactly what the benchmark + NCU must decide.
- Large-B prefill radix (~75 µs tail): expected **selection / store bound**. The candidate currently
  falls back to the baseline here (no native radix bucket), so no change is expected on these rows.

## Win/No-Go decision: UNDETERMINED (blocked)
- **Win** (AC-8) would require: production-row geomean > 1.0 for the native candidate vs the recovered
  baseline, measured in one fair strict-idle run, with full correctness (already have 251/251).
- **No-go** (AC-8) would require: baseline numbers + ≥1 reasoned attempt (have the candidate) +
  correctness status (have it) + **benchmark/NCU evidence** + a named active bound.
- Both require the candidate-vs-baseline benchmark (and, for the bound, NCU). Neither exists yet because
  GPU 1 is not strictly idle and the user chose to wait. **No speedup is claimed.**

## To finish (when GPU 1 is strictly idle, or the user authorizes Option B)
1. Re-check GPU 1 (summary + compute-apps `nvidia-smi`); proceed only if no active compute AND no
   meaningful non-task residency. Record before/after evidence in `docs/run_log.md`.
2. `CUDA_VISIBLE_DEVICES=1 python3 bench/correctness.py cuda:0` → `matched_ratio==1.0 (251/251)`.
3. `CUDA_VISIBLE_DEVICES=1 python3 bench/benchmark.py --device cuda:0 --workloads bench/workloads.json
   --out bench/results.jsonl` → require `passed==total==251` (re-freezes the baseline AND times the
   native candidate in one fair interleaved run; the per-workload speedup is candidate vs baseline).
4. NCU the native decode path (and any regime that decides the call) for the active bound.
5. Replace the provisional sections above with the measured per-shape/per-regime medians, the production
   geomean headline, the named active bound, and the win/no-go decision; fill measured per-bucket
   speedups in `docs/dispatch.md`; decide task10 (next bucket) from the evidence.
