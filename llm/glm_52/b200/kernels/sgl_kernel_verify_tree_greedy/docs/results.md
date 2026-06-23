# Results — `verify_tree_greedy` baseline vs candidate (NVIDIA B200)

## Headline

**Outcome: evidence-backed NO-GO** (the recovered baseline is retained).

- Correctness: **17/17 exact-match** — candidate == baseline == independent CPU oracle,
  bit-for-bit on `predicts`, `accept_index`, `accept_token_num` (incl. untouched/poisoned
  slots) across the upstream fixture, all 10 production shapes (5 seeds each), and 6
  regression rows (incl. the `bs>CAP, nd>2` baseline-fallback row).
- Performance: production equal-weight **geomean speedup = 0.9956** (candidate ≈ 0.4%
  *slower*); per-row range 0.955–1.007. **No statistically-clear win** — baseline and
  candidate p10/p90 bands overlap on every production row. Per DEC-1 this is a no-go.
- Active bound: **kernel-launch / device-scheduler overhead** (~4 µs/launch floor). The
  kernel does < 1 KB of work and sits at ~0.002% of the B200 bandwidth roofline; further
  kernel-body or launch-geometry work cannot move a launch-bound time.

Authoritative metric: CUDA-event GPU time (DEC-2). Environment, GPU-idle evidence,
versions, commands, and source hashes: see `docs/run_log.md`. Method: `docs/benchmark_method.md`.

## Per-shape results (amplified CUDA-event GPU time, µs; idle B200 GPU 7)

Inner-loop amplification = 256 launches/sample (256 × ~4 µs ≈ 1024 µs); 7 trials,
interleaved A/B, isolated subprocess. `speedup = baseline_median / candidate_median`.

| Workload | prod | base med | cand med | base p10/p90 | cand p10/p90 | speedup |
|----------|:----:|---------:|---------:|--------------|--------------|--------:|
| prod_bs1_nd2_ss2  | ✓ | 4.038 | 4.010 | 4.018 / 4.251 | 3.974 / 4.135 | 1.0070 |
| prod_bs2_nd2_ss2  | ✓ | 4.059 | 4.100 | 3.327 / 4.101 | 3.370 / 4.115 | 0.9900 |
| prod_bs3_nd2_ss2  | ✓ | 3.879 | 4.060 | 3.205 / 4.094 | 3.283 / 4.107 | 0.9554 |
| prod_bs4_nd2_ss2  | ✓ | 4.103 | 4.093 | 3.198 / 4.111 | 3.859 / 4.113 | 1.0025 |
| prod_bs5_nd2_ss2  | ✓ | 4.807 | 4.819 | 4.732 / 5.209 | 4.791 / 5.020 | 0.9976 |
| prod_bs6_nd2_ss2  | ✓ | 4.083 | 4.095 | 3.858 / 4.105 | 3.896 / 4.110 | 0.9970 |
| prod_bs7_nd2_ss2  | ✓ | 4.098 | 4.092 | 3.944 / 4.110 | 3.968 / 4.108 | 1.0015 |
| prod_bs8_nd2_ss2  | ✓ | 4.103 | 4.098 | 3.990 / 4.109 | 3.746 / 4.107 | 1.0013 |
| prod_bs9_nd2_ss2  | ✓ | 4.096 | 4.076 | 4.054 / 4.104 | 3.967 / 4.112 | 1.0051 |
| prod_bs10_nd2_ss2 | ✓ | 4.094 | 4.096 | 4.065 / 4.105 | 3.401 / 4.107 | 0.9995 |
| reg_upstream_nd6_ss4_bs2 | – | 4.111 | 4.113 | 4.108 / 4.114 | 4.111 / 4.115 | 0.9995 |
| reg_partial_accept_nd4_ss3_bs3 | – | 4.109 | 4.105 | 4.091 / 4.113 | 4.097 / 4.114 | 1.0009 |
| reg_full_reject_nd2_ss2_bs4 | – | 4.104 | 4.047 | 4.057 / 4.107 | 3.115 / 4.088 | 1.0140 |
| reg_full_accept_nd4_ss4_bs2 | – | 4.107 | 4.106 | 4.057 / 4.112 | 4.063 / 4.109 | 1.0001 |
| reg_sibling_tiebreak_nd5_ss3_bs1 | – | 4.111 | 4.109 | 4.104 / 4.115 | 4.106 / 4.112 | 1.0005 |
| reg_fallback_nd5_ss4_bs16 | – | 4.113 | 4.114 | 4.110 / 4.114 | 4.109 / 4.117 | 0.9996 |

**Production headline:** geomean 0.9956, arithmetic mean 0.9957, min 0.9554, max 1.0070.

## Decision (DEC-1, DEC-2)

The promotion rule is "promote on any statistically-clear win (production CUDA-event
p10/p90 bands separating cleanly from baseline, no production regression); otherwise an
evidence-backed no-go." Here:
- The geomean is **below 1.0** (0.9956) — the candidate is not faster on average.
- On **every** production row the baseline and candidate p10/p90 bands **overlap**; no row
  shows a clean separation favoring the candidate. The few rows >1.0 (bs1 1.007, bs9 1.005)
  are within the same noise that produces rows <1.0 (bs3 0.955).
→ No statistically-clear win. **No-go.**

## Active bound — roofline / launch analysis

Per call the kernel touches at most `bs·nd = 20` int64 candidate slots. Worst captured
case (bs=10): ~6 int64 loads + ~4 int32 stores per request ≈ **640 B total** (< 1 KB),
no floating point, a few dozen integer ops.

- B200 HBM bandwidth ≈ 8 TB/s ⇒ a bandwidth-bound 640 B transfer would take ≈ **0.00008 µs**.
  The measured ~4 µs/launch is **~50,000×** larger — the kernel sits at **~0.002%** of the
  bandwidth roofline (and similarly negligible against the integer-compute roofline).
- The measured ~4 µs is **invariant** to: batch size (bs 1→10), block count (baseline's
  `bs` blocks vs the candidate's single block), and tree size (regression rows nd 2→6,
  nss 2→4, bs→16 all sit at ~4.10–4.11 µs). Time that does not change with the amount of
  work or the launch geometry is, by definition, **fixed per-launch overhead**.

**Named active bound: kernel-launch / device-scheduler overhead (~4 µs floor on this
B200 / CUDA-13 / PyTorch-2.11 stack).** The kernel body is far below the
measurement/actionability threshold.

### Why the candidate cannot win

The only credible kernel-level lever for this op is launch geometry. The candidate applies
it — one thread per request in a single block, trimming the block count from `bs` to 1,
with `__ldg`/`__restrict__` read-only loads and compile-time `nd=2/nss=2` specialization.
The measured time is unchanged (~4 µs) because the launch floor dominates; no change to the
kernel body or geometry can reduce a launch-bound latency. (The baseline's `bs`-block and
the candidate's single-block geometries bracket the realistic design space and both land on
the same floor.) The **bs=1** row is especially diagnostic: there baseline and candidate
launch the *same* one block, and the times are still 4.038 vs 4.010 µs — i.e. even with
geometry held identical there is no body-level win to be had.

## Independent analysis (Codex, task11 `analyze`)

An independent Codex review of the data reached the same verdict: the dominant cost is the
"single CUDA kernel launch / device dispatch floor, not memory bandwidth, arithmetic
throughput, occupancy, or block scheduling"; observed effective bandwidth is ~0.25 GB/s
(< 1 KB / 4 µs), orders of magnitude below any B200 roofline; and **`NO_GO_JUSTIFIED: YES`**
because the no-go bar is met (recovered baseline, one credible candidate on the only
plausible lever, correctness vs baseline + CPU oracle, amplified interleaved CUDA-event
timing, named bound) while the geomean 0.9956 with fully overlapping p10/p90 bands means no
statistically-clear win. Codex agreed NCU is low-value and the launch-amortizing directions
(CUDA Graphs, persistent kernel, adjacent-kernel fusion, server-side batching) are
system-level and out of scope for a standalone single-launch benchmark.

## Dispatch table

The candidate is shape-specialized with a cheap host-side dispatcher (no device sync on the
hot path); both paths are correctness-verified. It is **not promoted** (no measurable
benefit), so the baseline remains the implementation.

| Shape regime | Selected path |
|--------------|---------------|
| `num_draft_tokens == 2 && num_spec_step == 2 && bs ≤ 1024` (captured GLM-5.2 regime) | specialized lane-per-request candidate |
| everything else (wider/deeper trees, larger batch) | recovered baseline (fallback) |

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
  work-/geometry-invariant ~4 µs floor are the diagnosis.
- **Warp-specialization**: not applicable — the candidate is a single-thread-per-request
  kernel with no producer/consumer warp roles, mbarrier, or pipeline; the
  warp-specialization-report-skill does not apply.

## Conclusion

`verify_tree_greedy` on the captured GLM-5.2 B200 shapes is **launch/scheduler-bound**: the
recovered baseline already runs at the ~4 µs per-launch floor and a correct, ABI-matched,
shape-specialized native-CUDA candidate cannot beat it. **Retain the upstream baseline.**
The candidate (correct, behind the same ABI, with baseline fallback) is kept in `solution/`
as a verified attempt but is not promoted.
