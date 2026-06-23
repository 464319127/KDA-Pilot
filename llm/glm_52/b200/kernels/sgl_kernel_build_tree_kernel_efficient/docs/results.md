# Results: `sgl_kernel.build_tree_kernel_efficient` (GLM-5.2, B200)

Run on remote host `ion-b200`, NVIDIA B200 GPU id 6 (idle before and after).
Upstream baseline commit `7e6587c94a1d0305815a14067c5d3cc02a9b0f36`. torch
2.11.0+cu130, nvcc 13.0, compiled for `sm_100`, symmetric `-O3` (no fast-math).

## Headline
- **Correctness: PASS** — `bench/correctness.py` ran **83 cases, 0 failures**.
  Baseline AND candidate both match an independent Python oracle bit-for-bit
  (exact int64/bool/tree) across uniform/skewed/monotonic/random
  `verified_seq_len` distributions, the full captured `(bs,T)` range sweep,
  poison + required pre-state, separate in-place copies, and both fallback rows.
  The official benchmark's inline exact-match gate also passed **15/15**.
- **Performance: small, real, launch-floor-bounded WIN.** Standard-harness
  production geomean = **1.144×**; a controlled same-process probe (31 trials,
  tight p10/p90) gives a conservative **1.021×** with non-overlapping p10/p90 for
  bs 2–10 and **no real regression**. Empty-kernel launch floor ≈ 2.0–3.2µs vs the
  op's ≈ 4µs ⇒ **~50–80% of the runtime is the irreducible CUDA launch floor**.
  Per DEC-2 ("any statistically stable, regression-free geomean win > 1.0") this
  is a (marginal) win; the candidate is promoted.

## Per-shape baseline-vs-candidate (official harness, CUDA-event median µs)
| id | bs | prod | baseline_us | candidate_us | speedup | inner (b/c) |
|---|---|---|---|---|---|---|
| glm52_bs1_T300 | 1 | true | 4.122 | 2.887 | 1.428 | 256/512 |
| glm52_bs2_T1234 | 2 | true | 4.120 | 2.879 | 1.431 | 256/512 |
| glm52_bs3_T3814 | 3 | true | 4.120 | 2.955 | 1.394 | 256/512 |
| glm52_bs4_T2246 | 4 | true | 4.114 | 3.036 | 1.355 | 256/512 |
| glm52_bs5_T5528 | 5 | true | 4.117 | 4.046 | 1.017 | 256/256 |
| glm52_bs6_T6216 | 6 | true | 5.272 | 5.264 | 1.001 | 256/256 |
| glm52_bs7_T6024 | 7 | true | 5.264 | 5.387 | 0.977* | 256/256 |
| glm52_bs8_T9138 | 8 | true | 4.119 | 4.064 | 1.014 | 256/256 |
| glm52_bs9_T9842 | 9 | true | 5.304 | 5.402 | 0.982* | 256/256 |
| glm52_bs10_T5382 | 10 | true | 4.128 | 4.093 | 1.008 | 256/256 |
| edge_minT | 1 | false | 4.115 | 2.837 | 1.451 | 256/512 |
| edge_maxT | 10 | false | 4.116 | 4.093 | 1.006 | 256/512 |
| edge_bs1_seq0 | 1 | false | 4.117 | 2.917 | 1.412 | 256/512 |
| fallback_qlen_only_bs4 | 4 | false | 4.115 | 4.130 | 0.996 | 256/256 |
| fallback_noncontig_bs4 | 4 | false | 4.518 | 4.698 | 0.962 | 256/256 |

- **Production geomean = 1.144×** (`headline.geomean_speedup`), arith-mean 1.161,
  min 0.977, max 1.431, n=10.
- `*` bs7 (0.977) and bs9 (0.982) are **subprocess clock artifacts**, not real
  regressions: in those isolated subprocesses BOTH sides ran ~5.3µs (a low-clock
  state) vs ~4.1µs elsewhere. A controlled same-process probe (both sides timed
  back-to-back, 31 trials) shows the candidate clean-faster (non-overlapping
  p10/p90) for bs 2–10 with geomean **1.021×** and **no regression**. The honest
  read: a small but real win, ~2% in a controlled clock state and up to ~1.4× for
  small bs when the harness happens to clock high.

## Active bound (roofline-style explanation)
This op is a tiny, single-launch index/mask kernel: for the captured fixed regime
it writes only O(bs) integer entries plus one `tree_mask` flip per request (a few
KB at most), far below any compute or DRAM-bandwidth limit on B200. Empty-kernel
floor (`build_tree_noop`, same grid/block, `bench/floor_probe.py`):
- bs=4: floor **2.06µs**, baseline 4.11µs, candidate 3.57µs (baseline/floor ≈ 2.0)
- bs=10: floor **2.04µs**, baseline 4.11µs, candidate 4.10µs (baseline/floor ≈ 2.0)
- (floor ≈ 3.18µs in a lower-clock state)

So **~50–80% of the op's ~4µs runtime is irreducible launch/scheduling latency**;
only the remaining ~1–2µs is addressable on the strict op ABI. The candidate
captures part of it (single block vs the baseline's `bs` blocks; no parent
traversal; minimal writes), most visibly at small bs. The active bound is
**kernel-launch / scheduling latency**, not compute or bandwidth. NCU on a ~4µs
launch-bound kernel would only confirm this (duration ≈ launch overhead,
negligible SM/DRAM utilization), so the floor measurement is the direct,
sufficient evidence and a separate NCU trace adds no actionable signal here.
Warp-specialization profiling does **not** apply (not a producer/consumer
pipelined GEMM); see `docs/dispatch.md`.

## Wrapper-inclusive diagnostic (DEC-1: secondary only, NOT promoted)
The captured op runs on outputs the higher-level SGLang Python wrapper
pre-allocates and pre-fills (`tree_mask.fill_(True)`, `retrieve_buf =
full((3,bs,nv), -1)`). Each prefill is itself a separate launch-bound kernel
(~floor latency). Inferred from the measured floor, the wrapper-inclusive path
(alloc + 1 mask fill + 1 retrieve fill + op) adds roughly two extra launch-floor
intervals on top of the op, i.e. the prefill is on the same order as the op
itself. The high-leverage optimization is therefore to **fuse the wrapper prefill
into the kernel** (eliminating those extra launches) — but that changes the
measured ABI boundary and is **out of promotion scope per DEC-1**. Recorded here
as an opt-in recommendation: a future wider integration patch could fuse the
prefill; it requires explicit user opt-in. (A precise wrapper-inclusive
measurement is a recommended follow-up for that patch.)

## Verdict
Correct, reproducible, and a small statistically-stable geomean win on the strict
captured op ABI, bounded by the CUDA launch floor (~50–80% of runtime). The
candidate is promoted (DEC-2 satisfied: geomean > 1.0, no real production
regression). The larger latency opportunity lives in the wrapper prefill, held
out of scope by DEC-1 and recorded as an opt-in recommendation.

## Independent cross-check (Codex, task12)
An independent Codex review of these results concurred: a **defensible marginal
promote** under DEC-2 (not a no-go) — correctness strong, geomean > 1.0, and the
paired same-process probe credibly attributes bs7/bs9 < 1.0 to subprocess clock
artifacts (both sides moved into the same ~5.3µs slow band). It agreed the
empty-kernel floor is stronger, sufficient evidence than an NCU trace for the
launch-bound question (NCU not required), and that warp-specialization does not
apply (no TMA/MMA producer-consumer pipeline; adding warp roles would only add
overhead). Its one caution — that the official 1.144× overstates practical
confidence and the controlled **1.021×** is the more honest claim — is reflected
in the headline above. Net guidance: promote narrowly as a regression-free,
launch-bound win with expected production gain in the low single digits; the real
roofline escape hatch is wrapper/prefill fusion (held out of scope by DEC-1).

## Provenance
- Host / GPU id 6 / model / before+after idle + versions: `docs/run_log.md`.
- Baseline commit + copied files: `docs/baseline_source.md`.
- Method / ABI / flags / timing / ring: `docs/benchmark_method.md`.
- Dispatch table + per-regime: `docs/dispatch.md`.
- Raw per-run records: `bench/results.jsonl` (kept local, gitignored).
