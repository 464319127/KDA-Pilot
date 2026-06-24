# Results — `moe_fused_gate` candidate vs baseline (B200)

## Headline
- **Correctness:** candidate passes the full grid — **829 checks, 0 failures** (`bench/correctness.py`,
  which loads `bench/workloads.json`): candidate-vs-**oracle** on **all 18 decode + 11 prefill**
  captured rows (×2 seeds) + boundary; candidate-vs-baseline + baseline-vs-oracle on prefill; the
  frozen **semantic edge grid** (all-equal, tie-smaller-index, saturate ±, **+Inf/-Inf**,
  subnormal-sum, M=0) built via the same `generator` field as `workloads.json`; **off-domain (E=256)
  fallback** (candidate==baseline==oracle); determinism. The candidate is **cold-safe** (correct as
  the first launch on a cold context). NB: the grid does **not** run the baseline decode path (UB);
  the decode reference is the independent oracle. The *warmed* baseline decode also matching the
  oracle (M=1/32/79/512) was a separate one-off probe and only shows state-dependent UB — not that
  the baseline decode is safe.
- **Performance (prefill, candidate vs baseline):** equal-weight (= call-count-weighted, since all
  prefill rows share call_count=456) **geometric-mean speedup = 1.0105** over the 11 prefill rows
  (range 0.982–1.135) — **parity within run-to-run noise** for a ~6–8 µs launch-bound kernel (the
  1.135 row is a variance outlier; Round 0 measured it at 1.022). No material repeatable regression
  or win.
- **Performance (decode):** the recovered **baseline decode path is unsafe (latent UB, see below)
  and not safely benchmarkable**; the candidate runs correctly at the kernel-launch floor (~4.1 µs
  at M=1), measured by the candidate-only `bench/bench_decode_candidate.py`. No speed ratio is
  reported for decode because the baseline cannot be reliably timed there.
- **Verdict (scoped to this standalone ABI + the captured MiniMax-M3 config):** **PROMOTE the
  candidate as a correctness + safety fix at performance parity.** A *pure-speed* optimization is an
  evidence-backed **NO-GO** for this standalone op: NCU shows ~0% compute and ~0% DRAM utilization,
  so it is launch/latency-bound and a standalone kernel (fixed ABI, no host-side batching/PDL) has
  no headroom to win. The candidate's value is that it is correct and cold-safe on the dominant
  captured decode config, where the baseline has a latent illegal-memory-access bug, with no
  material measured regression on the prefill rows (satisfies the directional + no-regression bar,
  DEC-1).

## Per-regime summary
| Regime | Rows | Baseline | Candidate | Result |
|---|---|---|---|---|
| Decode (M≤512, small-token) | 18 captured (M=1..79; `production:false` — unbenchmarkable) | **UB — nondeterministic illegal access; not A/B-benchmarkable** | correct + cold-safe, ~4.1–6.1 µs | correctness/safety **win** (categorical) |
| Prefill (M>512, large-token) | 11 production (M=1074..7432) | correct, ~6–8 µs | correct, ~6–8 µs | **parity** (geomean 1.0105, within noise) |

## Aggregate table (the decode baseline is UB, so production-wide/decode ratios are N/A)
| Aggregate | Value | Notes |
|---|---|---|
| Production-wide speedup (decode+prefill) | **N/A** | baseline decode path is UB/unbenchmarkable → invalid denominator |
| Decode speedup | **N/A** | baseline decode unbenchmarkable; candidate absolute latency reported below |
| Decode candidate absolute latency | 4.10 µs (M=1) … 6.14 µs (M=512) | `bench/bench_decode_candidate.py`; baseline cannot run there |
| Prefill equal-weight geomean | **1.0105** | 11 rows |
| Prefill call-count-weighted geomean | **1.0105** | identical — all prefill rows have equal call_count (456) |
| Prefill per-row range | 0.982 – 1.135 | spread is run-to-run variance on a ~6–8 µs launch-bound kernel |
| Decode share of production calls | 84.2% | 26670 / 31686 — the dominant regime, where the baseline is unsafe |

The prefill aggregate is **parity within noise** (geomean ≈ 1.01; the m4951 row's 1.135 is a
variance outlier on a sub-µs-body launch-bound kernel, not a repeatable win — Round 0 measured the
same row at 1.022). The candidate is neither materially faster nor slower than the baseline on
prefill.

## Per-shape table — prefill (candidate vs baseline, CUDA-event median, num_trials=9, --no-isolated, idle GPU 4)
| M (E=128, topk=5) | baseline µs | candidate µs | speedup |
|---|---|---|---|
| 1074 | 6.516 | 6.634 | 0.982 |
| 2340 | 6.533 | 6.655 | 0.982 |
| 4004 | 6.628 | 6.697 | 0.990 |
| 4339 | 6.621 | 6.673 | 0.992 |
| 4951 | 8.220 | 7.243 | 1.135 |
| 5398 | 8.252 | 8.182 | 1.009 |
| 5956 | 8.243 | 8.231 | 1.001 |
| 7120 | 8.279 | 8.241 | 1.005 |
| 7149 | 8.272 | 8.257 | 1.002 |
| 7299 | 8.438 | 8.260 | 1.022 |
| 7432 | 8.322 | 8.280 | 1.005 |
| **equal-weight geomean** | | | **1.0105** |

## Candidate decode latency (candidate-only; baseline decode is UB/unbenchmarkable)
| M | candidate µs/call |
|---|---|
| 1 | 4.10 |
| 7 | 4.10 |
| 16 | 4.35 |
| 32 | 4.35 |
| 79 | 4.61 |
| 512 | 6.14 |

(Inner-loop amplified, 9 trials, GPU 4 idle. These are at the kernel-launch floor — latency, not work, dominates.)

## Evidence-backed analysis (named active bound)
NCU (`--set basic`, candidate, GPU 4):

| Metric | Decode (M=1) | Prefill (M=7432) |
|---|---|---|
| Compute (SM) throughput | 0.06 % | 0.06 % |
| Memory throughput | 2.93 % | 3.08 % |
| DRAM throughput | 0.02 % | 0.03 % |
| Achieved occupancy | 12.5 % | 13.0 % |

**Active bound: kernel-launch / device-scheduling latency.** Compute and DRAM utilization are
effectively zero; the kernel moves trivial data (≈ `M·128·4 B` in, `M·5·(4+4) B` out) and does a
handful of `sigmoid`+top-5 ops per token — arithmetic intensity is negligible and at M=1 the
useful traffic (~512 B) over ~4 µs is ~0.1 GB/s, ~5 orders of magnitude below B200 HBM. There is
no compute- or bandwidth-bound region to optimize; the only lever is launch overhead, which a
single standalone op (fixed ABI, no host-side batching/PDL) cannot reduce. This matches the
first-pass prediction that the op is launch-bound and a speed win is implausible.

**Warp-specialization-report-skill: NOT APPLICABLE.** The candidate is a single-pass
warp-per-token kernel with no producer/consumer pipeline, no TMA/TMEM staging, and no
mbarrier/named-barrier coordination, so there is nothing for a warp-specialization timeline to
reconcile. (Recorded per the task requirement.)

## Baseline decode UB (key correctness finding)
The recovered upstream jit_kernel `moe_fused_gate` small-token (decode) kernel reads
**uninitialized shared memory** for `num_experts=128`: it computes `warps_per_token =
div_ceil(128,32) = 4` yet is dispatched to the `<8>`-warp template, so warp 0's final cross-warp
reduction reads the never-written `warp_maxs[4..7]` / `warp_experts[4..7]`; when an uninitialized
slot exceeds the real max, a garbage expert index can propagate to an out-of-bounds
`shared_scores[selected] = -FLT_MAX` write. On B200/GPU 4 the decode path
**raised `CUDA error: an illegal memory access` on some cold-context runs and ran (matching the
oracle) on others** — classic nondeterministic uninitialized-memory UB. The prefill (large-token)
path has no such read and matches the oracle exactly. The candidate avoids the bug entirely by
using **zero shared memory** (all reductions are intra-warp `__shfl`), so it is correct and safe as
the first launch on a cold context. Full analysis in `docs/baseline_source.md`.

## Correctness
- 829/829 checks pass (see `bench/correctness.py`, which loads `bench/workloads.json`). Indices exact-match the **oracle** on all paths;
  weights within `atol=rtol=1e-5`. Adversarial ties resolve to the smaller index on both paths;
  subnormal-stress and M=0 covered. The grid does NOT run the baseline decode path (UB); the oracle
  is the decode reference. The warmed-baseline-decode==oracle result (M=1/32/79/512) came from a
  separate probe and only shows state-dependent UB.
- Off-domain (E=256, topk=8) is exercised: the candidate routes to the verbatim-copied baseline
  fallback and is candidate==baseline==oracle (small- and large-token). The fallback is bit-identical
  to the baseline (incl. its UB for off-domain E≤224 small-token) — see `docs/dispatch.md` for the
  scoped safety statement.
- Input contract: all 296 captures are finite fp32. **+Inf/-Inf are covered** as explicit edge rows
  (`pos_inf`/`neg_inf`; sigmoid(±inf) is finite 1/0, so candidate and oracle agree). **Only NaN is
  out of contract** (baseline ignores NaN in `>` reductions; the candidate's packed key may order it
  differently) — see `docs/benchmark_method.md`.

## Provenance
- GPU: NVIDIA B200, host `ion-b200`, `REMOTE_GPU_ID=4`; idle before measurement (0%/0 MiB), only
  this benchmark's own process active during (see `docs/run_log.md`).
- Baseline: SGLang `main` @ `34dd9c28caf4f7dd185e58e462a1344b52568e2e` (jit_kernel header).
- Software: PyTorch 2.11.0+cu130, CUDA 13.0, nvcc 13.0, tvm_ffi; symmetric build flags both sides.
- Harness: `bench/benchmark.py` verbatim from the project template; CUDA-event timing, inner-loop
  amplification, interleaved A/B, num_trials≥7.

## Conclusion
For the captured MiniMax-M3 config, the candidate is a **correct, cold-safe, native-CUDA drop-in**
that **eliminates the baseline's decode-path UB for that config** at **performance parity** on
prefill (geomean 1.0105, within run-to-run noise; no material repeatable regression) and at the launch-latency floor on decode.
**PROMOTE** on correctness/safety grounds. As a *speed* optimization the task is a well-evidenced
**NO-GO** for this standalone op — it is launch/latency-bound (NCU ~0% compute / ~0% DRAM), so no
standalone-kernel speedup is achievable under the fixed ABI; this conclusion rests on recovered
baseline numbers, a reasoned native-CUDA candidate, full correctness (829 checks), benchmark deltas,
NCU evidence, and a named active bound (launch/scheduling latency). Scope: the candidate is not a
general fix for the upstream `moe_fused_gate` bug on arbitrary off-domain E=128 configs (see
`docs/dispatch.md`).
