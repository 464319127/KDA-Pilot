# Results — `moe_fused_gate` candidate vs baseline (B200)

## Headline
- **Correctness:** candidate passes the full grid — **482 checks, 0 failures** (candidate-vs-oracle on every decode+prefill+boundary shape; candidate-vs-baseline on prefill; ties/edges/M=0/determinism). The candidate is **cold-safe** (runs correctly as the first launch on a cold context).
- **Performance (prefill, candidate vs baseline):** equal-weight **geometric-mean speedup = 1.0006** over the 11 prefill production rows (range 0.992–1.022) — **parity, no regression**.
- **Performance (decode):** the recovered **baseline decode path is unsafe (latent UB, see below) and not benchmarkable**; the candidate runs correctly at the kernel-launch floor (~4.1 µs at M=1). No fair speed ratio exists because the baseline cannot be reliably timed there.
- **Verdict:** **PROMOTE the candidate as a correctness + safety fix at performance parity.** A *pure-speed* optimization is an evidence-backed **NO-GO**: the op is launch/latency-bound with ~0% compute and ~0% DRAM utilization, so a standalone kernel has no headroom to win. The candidate's value is that it is correct and cold-safe on the dominant decode regime where the baseline has a latent illegal-memory-access bug, with zero regression elsewhere (satisfies the directional + no-regression bar, DEC-1).

## Per-regime summary
| Regime | Rows | Baseline | Candidate | Result |
|---|---|---|---|---|
| Decode (M≤512, small-token) | 18 production (M=1..79) | **UB — nondeterministic illegal access; not benchmarkable** | correct + cold-safe, ~4.1–6.1 µs | correctness/safety **win** (categorical) |
| Prefill (M>512, large-token) | 11 production (M=1074..7432) | correct, ~6–8 µs | correct, ~6–8 µs | **parity** (geomean 1.0006, no regression) |

## Per-shape table — prefill (candidate vs baseline, CUDA-event median, num_trials=9, --no-isolated)
| M (E=128, topk=5) | baseline µs | candidate µs | speedup |
|---|---|---|---|
| 1074 | 5.802 | 5.846 | 0.992 |
| 2340 | 6.007 | 6.043 | 0.994 |
| 4004 | 6.520 | 6.560 | 0.994 |
| 4339 | 7.283 | 7.328 | 0.994 |
| 4951 | 8.217 | 8.036 | 1.022 |
| 5398 | 8.239 | 8.234 | 1.001 |
| 5956 | 8.234 | 8.238 | 0.999 |
| 7120 | 8.243 | 8.240 | 1.000 |
| 7149 | 8.243 | 8.243 | 1.000 |
| 7299 | 8.309 | 8.239 | 1.008 |
| 7432 | 8.256 | 8.243 | 1.002 |
| **geomean** | | | **1.0006** |

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
reduction reads the never-written `warp_maxs[4..7]` / `warp_experts[4..7]`; a garbage expert index
becomes an out-of-bounds `shared_scores[selected] = -FLT_MAX` write. On B200/GPU 4 the decode path
**raised `CUDA error: an illegal memory access` on some cold-context runs and ran (matching the
oracle) on others** — classic nondeterministic uninitialized-memory UB. The prefill (large-token)
path has no such read and matches the oracle exactly. The candidate avoids the bug entirely by
using **zero shared memory** (all reductions are intra-warp `__shfl`), so it is correct and safe as
the first launch on a cold context. Full analysis in `docs/baseline_source.md`.

## Correctness
- 482/482 checks pass (see `bench/correctness.py`). Indices exact-match the oracle on all paths;
  weights within `atol=rtol=1e-5`. Adversarial ties resolve to the smaller index on both paths.
  Independent oracle validated against the warmed baseline (decode + prefill).
- Off-domain inputs (E≠128, other params, non-fp32) route to the verbatim-copied baseline fallback
  (bit-identical), via a host-side gate with no device sync — see `docs/dispatch.md`.

## Provenance
- GPU: NVIDIA B200, host `ion-b200`, `REMOTE_GPU_ID=4`; idle before measurement (0%/0 MiB), only
  this benchmark's own process active during (see `docs/run_log.md`).
- Baseline: SGLang `main` @ `34dd9c28caf4f7dd185e58e462a1344b52568e2e` (jit_kernel header).
- Software: PyTorch 2.11.0+cu130, CUDA 13.0, nvcc 13.0, tvm_ffi; symmetric build flags both sides.
- Harness: `bench/benchmark.py` verbatim from the project template; CUDA-event timing, inner-loop
  amplification, interleaved A/B, num_trials≥7.

## Conclusion
The candidate is a **correct, cold-safe, native-CUDA drop-in** that **eliminates the baseline's
decode-path UB** at **performance parity** on prefill (geomean 1.0006, no regression) and at the
launch-latency floor on decode. **PROMOTE** on correctness/safety grounds. As a *speed*
optimization the task is a well-evidenced **NO-GO** — the op is launch/latency-bound (NCU ~0%
compute / ~0% DRAM), so no standalone-kernel speedup is achievable; this conclusion rests on
recovered baseline numbers, a reasoned native-CUDA candidate, full correctness, benchmark deltas,
NCU evidence, and a named active bound (launch/scheduling latency).
