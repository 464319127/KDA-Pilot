# Results — `verify_tree_greedy` baseline vs candidate (NVIDIA B200)

Round 1 run through the literal **TVM-FFI direct-symbol ABI** (re-implemented per the
Round-0 review; correctness + benchmark re-measured). The verdict is unchanged.

## Headline

**Outcome: evidence-backed NO-GO** (the recovered baseline is retained).

- Correctness: **17/17 exact-match** — candidate == baseline == independent CPU oracle,
  bit-for-bit on `predicts`, `accept_index`, `accept_token_num` (incl. untouched/poisoned
  slots) across the upstream fixture, all 10 production shapes (5 seeds each), and 6
  regression rows (incl. the `nd>2` baseline-fallback row).
- Performance: production equal-weight **geomean speedup = 0.9924** (candidate ≈ 0.8%
  *slower*); per-row range 0.977–1.012. **No statistically-clear win** — baseline and
  candidate p10/p90 bands overlap on every production row, and several rows are within-noise
  regressions. Per DEC-1 this is a no-go.
- Active bound: **kernel-launch / device-scheduler overhead** (~5 µs/launch floor this run).
  The kernel does < 1 KB of work and sits well under 0.01% of the B200 bandwidth roofline;
  further kernel-body or launch-geometry work cannot move a launch-bound time.

Authoritative metric: CUDA-event GPU time (DEC-2). Environment, GPU-idle evidence,
versions, commands, and source hashes: see `docs/run_log.md`. Method: `docs/benchmark_method.md`.
Dispatch buckets: `docs/dispatch.md`.

## Per-shape results (amplified CUDA-event GPU time, µs; idle B200 GPU 7)

Inner-loop amplification = 256 launches/sample (256 × ~5 µs ≈ 1280 µs); 7 trials,
interleaved A/B, isolated subprocess. `speedup = baseline_median / candidate_median`.

| Workload | prod | base med | cand med | base p10/p90 | cand p10/p90 | speedup |
|----------|:----:|---------:|---------:|--------------|--------------|--------:|
| prod_bs1_nd2_ss2  | ✓ | 5.005 | 5.069 | 4.935 / 5.213 | 4.983 / 5.233 | 0.9874 |
| prod_bs2_nd2_ss2  | ✓ | 5.576 | 5.688 | 5.553 / 5.778 | 5.581 / 5.990 | 0.9803 |
| prod_bs3_nd2_ss2  | ✓ | 5.961 | 6.018 | 5.896 / 6.016 | 5.989 / 6.074 | 0.9905 |
| prod_bs4_nd2_ss2  | ✓ | 4.883 | 4.998 | 4.858 / 5.160 | 4.954 / 5.327 | 0.9770 |
| prod_bs5_nd2_ss2  | ✓ | 5.381 | 5.404 | 5.233 / 5.655 | 5.171 / 5.449 | 0.9958 |
| prod_bs6_nd2_ss2  | ✓ | 5.795 | 5.852 | 5.767 / 5.836 | 5.758 / 6.018 | 0.9902 |
| prod_bs7_nd2_ss2  | ✓ | 5.094 | 5.032 | 4.994 / 5.137 | 5.013 / 5.604 | 1.0122 |
| prod_bs8_nd2_ss2  | ✓ | 6.140 | 6.094 | 6.120 / 6.323 | 5.999 / 6.236 | 1.0076 |
| prod_bs9_nd2_ss2  | ✓ | 4.974 | 4.989 | 4.888 / 5.164 | 4.911 / 5.124 | 0.9970 |
| prod_bs10_nd2_ss2 | ✓ | 5.682 | 5.763 | 5.608 / 6.048 | 5.681 / 6.008 | 0.9860 |
| reg_upstream_nd6_ss4_bs2 | – | 4.947 | 4.990 | 4.899 / 5.104 | 4.935 / 5.164 | 0.9913 |
| reg_partial_accept_nd4_ss3_bs3 | – | 5.651 | 5.710 | 5.601 / 5.793 | 5.569 / 6.025 | 0.9896 |
| reg_full_reject_nd2_ss2_bs4 | – | 5.082 | 5.060 | 5.034 / 5.291 | 5.025 / 5.262 | 1.0044 |
| reg_full_accept_nd4_ss4_bs2 | – | 5.138 | 5.236 | 5.068 / 5.432 | 5.092 / 5.382 | 0.9814 |
| reg_sibling_tiebreak_nd5_ss3_bs1 | – | 5.044 | 4.967 | 4.909 / 5.070 | 4.940 / 5.080 | 1.0155 |
| reg_fallback_nd5_ss4_bs16 | – | 4.848 | 4.879 | 4.801 / 5.070 | 4.826 / 5.126 | 0.9936 |

**Production headline:** geomean 0.9924, arithmetic mean 0.9924, min 0.9770, max 1.0122.

> Absolute floor note: this TVM-FFI run landed at ~4.8–6.1 µs/launch, higher than the
> Round-0 run (~4.0–4.8 µs) because other tenants' GPUs were busy during the run (host-level
> contention; GPU 7 itself stayed idle — see `docs/run_log.md`). Baseline and candidate are
> sampled **interleaved within each trial**, so both see identical conditions and the
> speedup *ratio* is unaffected. The floor rising with host contention (GPU-7 work unchanged)
> is itself further evidence the kernel is launch/host-bound, not body-bound.

## Decision (DEC-1, DEC-2)

The promotion rule is "promote on any statistically-clear win (production CUDA-event
p10/p90 bands separating cleanly from baseline, no production regression); otherwise an
evidence-backed no-go." Here:
- The geomean is **below 1.0** (0.9924) — the candidate is not faster on average, and 7 of
  10 production rows are within-noise regressions.
- On **every** production row the baseline and candidate p10/p90 bands **overlap**; no row
  shows a clean separation favoring the candidate. The rows >1.0 (bs7 1.012, bs8 1.008) sit
  in the same noise band as the rows <1.0 (bs4 0.977, bs10 0.986).
→ No statistically-clear win. **No-go.**

## Active bound — roofline / launch analysis

Per call the kernel touches at most `bs·nd = 20` int64 candidate slots. Worst captured
case (bs=10): ~6 int64 loads + ~4 int32 stores per request ≈ **640 B total** (< 1 KB),
no floating point, a few dozen integer ops.

- B200 HBM bandwidth ≈ 8 TB/s ⇒ a bandwidth-bound 640 B transfer would take ≈ **0.00008 µs**.
  The measured ~5 µs/launch is **~60,000×** larger — the kernel sits at **~0.0016%** of the
  bandwidth roofline (and similarly negligible against the integer-compute roofline).
- The measured per-launch time is **invariant** to: batch size (bs 1→10), block count
  (baseline's `bs` blocks vs the candidate's single block), and tree size (regression rows
  nd 2→6, nss 2→4, bs→16 all sit in the same ~4.8–6.1 µs band). It does, however, move with
  *host-level* contention (Round 0 ~4 µs on a quiet node vs ~5 µs here). Time that tracks
  launch/host overhead and ignores the amount of GPU work is, by definition, **fixed
  per-launch overhead**.

**Named active bound: kernel-launch / device-scheduler overhead (~5 µs floor this run on
this B200 / CUDA-13 / PyTorch-2.11 / tvm-ffi-0.1.9 stack).** The kernel body is far below
the measurement/actionability threshold.

### Why the candidate cannot win

The only credible kernel-level lever for this op is launch geometry. The candidate applies
it — one thread per request in a single block, trimming the block count from `bs` to 1,
with `__ldg`/`__restrict__` read-only loads and compile-time `nd=2/nss=2` specialization.
The measured time does not improve because the launch floor dominates; no change to the
kernel body or geometry can reduce a launch-bound latency. (The baseline's `bs`-block and
the candidate's single-block geometries bracket the realistic design space and both land on
the same floor.) The **bs=1** row is especially diagnostic: there baseline and candidate
launch the *same* one block, and the times are still 5.005 vs 5.069 µs — i.e. even with
geometry held identical there is no body-level win to be had.

## Independent analysis (Codex, task11 `analyze`)

An independent Codex review reached the same verdict on the (ABI-independent) launch-bound
regime: the dominant cost is the "single CUDA kernel launch / device dispatch floor, not
memory bandwidth, arithmetic throughput, occupancy, or block scheduling"; observed effective
bandwidth is a tiny fraction of a GB/s (< 1 KB per multi-µs launch), orders of magnitude
below any B200 roofline; and **`NO_GO_JUSTIFIED: YES`** because the no-go bar is met
(recovered baseline, one credible candidate on the only plausible lever, correctness vs
baseline + CPU oracle, amplified interleaved CUDA-event timing, named bound) while a
sub-1.0 geomean with fully overlapping p10/p90 bands means no statistically-clear win. Codex
agreed NCU is low-value and the launch-amortizing directions (CUDA Graphs, persistent
kernel, adjacent-kernel fusion, server-side batching) are system-level and out of scope for
a standalone single-launch benchmark. (The ABI change from Round 0 does not affect this
analysis — it changes neither the kernel nor the launch floor.)

## Dispatch table

The candidate is shape-specialized with a cheap host-side dispatcher (no device sync on the
hot path); both paths are correctness-verified. It is **not promoted** (no measurable
benefit), so the baseline remains the implementation. Full detail in `docs/dispatch.md`.

| Shape regime | Selected path |
|--------------|---------------|
| `num_draft_tokens == 2 && num_spec_step == 2` (captured GLM-5.2 family; production bs≤10, any bs supported) | specialized lane-per-request candidate |
| everything else (`nd≠2` or `nss≠2`: wider/deeper trees) | recovered baseline (fallback) |

## Residual directions (out of scope)

The launch floor can only be attacked *above* this kernel — e.g. CUDA Graphs capture,
a persistent/megakernel that fuses the verify step with neighboring speculative-decoding
work, or batching launches at the serving layer. All of these change the call contract /
ABI and live in the SGLang server integration, not in a standalone single-kernel-launch
benchmark. They are explicitly **out of scope** for this task and are noted for the serving
team rather than pursued here.

## Profiling notes

- **NCU**: not run — low value for a sub-1-KB, launch-bound integer kernel (bandwidth,
  FLOP/s, occupancy, and SM-utilization metrics are misleading when the kernel is dominated
  by fixed launch overhead and does negligible work). The roofline argument above and the
  work-/geometry-invariant per-launch floor are the diagnosis.
- **Warp-specialization**: not applicable — the candidate is a single-thread-per-request
  kernel with no producer/consumer warp roles, mbarrier, or pipeline; the
  warp-specialization-report-skill does not apply.

## Conclusion

`verify_tree_greedy` on the captured GLM-5.2 B200 shapes is **launch/scheduler-bound**: the
recovered baseline already runs at the per-launch floor and a correct, ABI-matched (literal
TVM-FFI direct-symbol), shape-specialized native-CUDA candidate cannot beat it. **Retain the
upstream baseline.** The candidate (correct, behind the same ABI, with baseline fallback) is
kept in `solution/` as a verified attempt but is not promoted.
