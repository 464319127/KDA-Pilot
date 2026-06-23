# Dispatch — `verify_tree_greedy` candidate

The candidate (`candidate_verify_tree_greedy`) performs a cheap, host-side dispatch that
reads only tensor metadata (shape/dtype) — **no device synchronization on the hot path** —
and routes each call to one of two paths. Source:
`solution/verify_tree_greedy_candidate.cuh` (`candidate_vtg::dispatch_verify_tree_greedy`).

## Buckets

| Bucket | Condition | Entry point | Launch geometry |
|--------|-----------|-------------|-----------------|
| Specialized (captured family) | `num_draft_tokens == 2 && num_spec_step == 2` | `candidate_vtg::VerifyTreeGreedyLaneW2S2` | one thread per request, `grid = ceil(bs / 128)`, `block = 128` (captured production `bs ≤ 10` ⇒ a single block) |
| Fallback | anything else (`nd ≠ 2` or `nss ≠ 2`) | `baseline_vtg::launch_baseline` (recovered upstream) | `grid(bs)`, `block(1)` |

Notes:
- The captured GLM-5.2 production regime is exactly `nd=2, nss=2, bs ∈ {1..10}`, so every
  production row takes the specialized path. The specialized path is correct for the whole
  `nd==2 && nss==2` family at any batch size (the lane-per-request kernel scales via the
  grid), so there is **no batch-size cap** — `bs` is just the grid dimension.
- Every shape outside that family (wider/deeper trees) falls back to the recovered baseline,
  so correctness is never lost on uncovered shapes. The candidate path then *is* the baseline
  code, by construction.

## Per-bucket measured behavior (B200 GPU 7; see `docs/results.md`, `docs/run_log.md`)

| Bucket | Workloads | Result |
|--------|-----------|--------|
| Specialized | `prod_bs1..10` (+ `reg_full_reject_nd2_ss2_bs4`) | production equal-weight geomean speedup = **0.9924** (range 0.977–1.012); all p10/p90 bands overlap baseline — **no statistically-clear win** |
| Fallback | `reg_upstream_nd6_ss4_bs2`, `reg_partial_accept_nd4_ss3_bs3`, `reg_full_accept_nd4_ss4_bs2`, `reg_sibling_tiebreak_nd5_ss3_bs1`, `reg_fallback_nd5_ss4_bs16` | candidate routes to the baseline launcher, so candidate ≡ baseline; measured ratios 0.98–1.02 are baseline-vs-baseline noise, and outputs match the baseline AND the oracle exactly |

Correctness across both buckets: **17/17 exact-match** (candidate == baseline == independent
oracle) — the fallback rows specifically verify that the dispatcher hands uncovered shapes to
the baseline and still produces bit-exact results.

## Rationale: one specialized variant, not more

Per `llm_kernel_optimization_rules.md` ("Shape Specialization"), additional specialized
kernels/template variants are warranted only when measured evidence shows different workload
buckets need different block sizes/layouts/register tradeoffs. Here the measured per-launch
time is **invariant** across the entire captured `bs` range (1→10) and across tree sizes —
every shape sits at the launch/scheduler floor (~5 µs this run; see the roofline analysis in
`docs/results.md`). There is no `bs` (or shape) regime where one configuration wins while
another loses, so a single specialized kernel plus baseline fallback is the right scope;
adding per-`bs` variants would be over-engineering with no measurable benefit.

Because the specialized path shows no statistically-clear win, the candidate is **not
promoted** — the recovered baseline remains the implementation (evidence-backed no-go).
