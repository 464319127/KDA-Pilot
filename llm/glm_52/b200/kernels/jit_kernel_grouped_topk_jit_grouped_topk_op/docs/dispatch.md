# Dispatch Table

The candidate (`solution/csrc/moe/grouped_topk_candidate.cuh`) selects a kernel at
launch time by inspecting `num_tokens` (N) and `num_experts` (E). The decision is
host-side only — no host sync on the hot path. Every supported input
(`num_expert_group=1, topk_group=1, topk<=8, num_experts<=512`) is covered;
unsupported parameters are rejected exactly as the baseline (same `RuntimeCheck`).

## Fast-path domain gate (checked first)

The native warp-per-token fast path runs **only** on the captured production
domain. The predicate is:

```
num_experts == 256 && topk == 8 && num_expert_group == 1 && topk_group == 1
  && renormalize == true && scaling_factor == 1.0 && num_tokens >= 768
```

Contiguity and fp32 are enforced earlier by the `TensorMatcher` verifies (a
non-contiguous or non-fp32 input is rejected identically to the baseline before
dispatch), so they are not re-tested in the predicate. **Any input that does not
satisfy the predicate falls back to the recovered baseline kernel**
(`grouped_topk_block_per_token_kernel`), which covers the entire baseline-supported
domain (`num_expert_group=1, topk_group=1, topk<=8, num_experts<=512`, any
`renormalize`/`scaling_factor`). Inputs the baseline itself rejects
(`num_expert_group≠1`, `topk_group≠1`, `topk>8`, `num_experts>512`) are rejected by
the same `RuntimeCheck`s on both sides.

## Buckets

| Case | Condition | Kernel | warps/block | Reason |
|---|---|---|---|---|
| **fallback** | not in fast-path domain (incl. `N < 768`, `E≠256`, `topk≠8`, `renormalize=False`, `scaling≠1`) | `grouped_topk_block_per_token_kernel` (recovered baseline algorithm) | n/a (E-tier threads, 1 block/token) | Decode/small-N is launch-floor / SFU-latency bound (full-block parallel sigmoid + per-token CTA spread beats warp-per-token); off-production cases are not the tuning target, so they take the proven baseline path. The candidate equals the baseline here (no regression, exact match). |
| fast path | production domain, `768 <= N <= 1280` | `grouped_topk_warp_per_token_kernel` | 8 (8 tokens/CTA) | Enough work to fill SMs; packing 8 tokens/CTA cuts block count at the transition. |
| fast path | production domain, `N > 1280` | `grouped_topk_warp_per_token_kernel` | 4 (4 tokens/CTA) | Best across the large region in the sweep; raises achieved occupancy vs the baseline while keeping enough blocks to avoid heavy wave-quantization tails. |

`production_domain` is a **hard gate**: the `K09_WPB` env override (1/2/4/8, tuning
sweeps only) can only change warps-per-block or lower the token threshold **within**
the production domain (E=256, topk=8, group 1/1, renormalize, scale=1.0). It can
**never** route an off-production input (E≠256, topk≠8, renormalize=False,
scaling≠1) to the warp kernel — those always fall back to the baseline kernel,
override or not. (The warp kernel keeps E-tier register sizing `ceil(E/32)` only so
the template is well-formed; the default and override dispatch both send only
E=256 to it.) A correctness regression (`bench/correctness.py`, run in a fresh
process with `K09_WPB=4`) asserts off-domain large-N still takes the baseline path.

## Per-bucket measured speedup (authoritative R3 run: idle GPU 6, `bench/results.jsonl`; baseline_median / candidate_median)

| Regime | N range | speedup | note |
|---|---|---|---|
| decode | 2–38 | **1.000** (parity) | identical kernel (copied baseline); launch-floor bound (~6.15 µs both) |
| mid | 110 | 1.009 | |
| small prefill | 392 / 445 / 489 / 645 | 1.081 / 1.041 / 1.040 / 1.000 | baseline-path region (N<768), near floor |
| prefill transition | 861–1167 | ~1.000 | warp path ties baseline (~8.20 µs) |
| prefill | 1464–1731 | **1.499** | warp path, baseline ~12.3 µs vs candidate ~8.2 µs |
| prefill | 1811–2247 | 1.20–1.33 | wave-quantization step in candidate time |
| prefill | 2798 | **1.598** | |
| prefill | 3617–3769 | **1.67** | max win; baseline ~20.5 µs vs candidate ~12.3 µs |

(Authoritative measured on idle GPU 6; the round-0 quiet-box GPU-0 run reproduced the
same pattern: decode 0.999, prefill up to 1.67×.)

## How the thresholds were chosen

A per-N sweep of warps/block ∈ {1,2,4,8} (see `run_log.md`) showed:
- For `N <= ~645` the warp-per-token path loses at every warps/block (0.7–0.92×) —
  one warp computing 8 serial `__expf` per lane is SFU-latency bound and a too-small
  grid underfills the 148-SM B200. So the baseline block-per-token kernel is kept
  below `N=768`.
- For `N >= ~861` the warp path reaches parity and then wins as N grows; fewer
  blocks per token (larger warps/block) helps the transition, while the very large
  region is dominated by per-token throughput and tolerates `W=4`.

The residual sawtooth in the 1811–2366 band is grid/SM **wave quantization** (block
count crossing an integer multiple of 148 SMs), an inherent GPU effect, not a
correctness or dispatch error. Further per-N warps/block tuning could smooth it but
yields marginal gains and risks overfitting to measurement noise.
