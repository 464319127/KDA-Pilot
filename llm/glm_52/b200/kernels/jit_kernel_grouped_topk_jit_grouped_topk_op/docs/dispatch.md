# Dispatch Table

The candidate (`solution/csrc/moe/grouped_topk_candidate.cuh`) selects a kernel at
launch time by inspecting `num_tokens` (N) and `num_experts` (E). The decision is
host-side only — no host sync on the hot path. Every supported input
(`num_expert_group=1, topk_group=1, topk<=8, num_experts<=512`) is covered;
unsupported parameters are rejected exactly as the baseline (same `RuntimeCheck`).

## Buckets (E = 256, topk = 8 — the production domain)

| Token regime | Condition | Kernel | warps/block | Reason |
|---|---|---|---|---|
| decode + small/mid prefill | `N < 768` | `grouped_topk_block_per_token_kernel` (recovered baseline algorithm) | n/a (256 threads, 1 block/token) | Launch-floor / SFU-latency bound; full-block parallel sigmoid + per-token CTA spread beats warp-per-token. The candidate equals the baseline here (no regression). |
| large prefill | `768 <= N <= 1280` | `grouped_topk_warp_per_token_kernel` | 8 (8 tokens/CTA) | Enough work to fill SMs; packing 8 tokens/CTA cuts block count at the transition. |
| large prefill | `N > 1280` | `grouped_topk_warp_per_token_kernel` | 4 (4 tokens/CTA) | Best across the large region in the sweep; raises achieved occupancy vs the baseline while keeping enough blocks to avoid heavy wave-quantization tails. |

Expert-count tiers (register slots per lane = `ceil(E/32)`): `E<=128 → 4`,
`E<=256 → 8` (production), `E<=512 → 16`. `K09_WPB` env var (1/2/4/8) forces the
warp path with that warps-per-block for tuning sweeps; unset = the table above.

## Per-bucket measured speedup (B200, idle GPU 0; baseline_median / candidate_median)

| Regime | N range | speedup | note |
|---|---|---|---|
| decode | 2–38 | **0.999** (parity) | identical kernel; launch-floor bound (~6.15 µs both) |
| mid | 110 | 1.039 | |
| small prefill | 392–645 | 1.02–1.05 | baseline-path region, near floor |
| prefill transition | 861–1167 | ~1.000 | warp path ties baseline (8.20 µs) |
| prefill | 1464–1731 | **1.50** | warp path, baseline at 12.3 µs vs candidate 8.2 µs |
| prefill | 1811–2366 | 1.20–1.28 | wave-quantization step in candidate time |
| prefill | 2798–3524 | **1.60** | |
| prefill | 3617–3769 | **1.67** | max win; baseline 20.5 µs vs candidate 12.3 µs |

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
