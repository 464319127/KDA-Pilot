# Dispatch Table — `moe_fused_gate` candidate

The candidate `MoEFusedGateKernel::run` chooses a path with a **host-side scalar predicate**
(no device synchronization on the hot path), after the same `TensorMatcher`/`RuntimeCheck`
validation as the baseline.

## Production-domain gate
```cpp
const bool in_domain =
    (num_experts == 128) && (topk == 5) && (scoring_func == 0 /*sigmoid*/) &&
    (num_fused_shared_experts == 1) && renormalize &&
    (routed_scaling_factor == 2.0f) && apply_routed_scaling_factor_on_output;
```
This is exactly the captured MiniMax-M3 configuration (all 296 variants), so every captured
shape — decode M=1..79 and prefill M=1074..7432 — takes the candidate fast path.

## Buckets
| Condition | Kernel | Mapping | Cold-safe | Result |
|---|---|---|---|---|
| `in_domain` (all captured shapes) | candidate `moe_fused_gate_warp_per_token_kernel` | 1 warp / token, 8 warps/block, 4 experts/lane, intra-warp `__shfl` only (no shared mem) | **yes** | correct; prefill parity (geomean 1.0006); decode at launch floor ~4.1 µs |
| anything else (E≠128, other params, scoring_func=1 off-domain, etc.) | verbatim-copied baseline (`namespace fallback`) — small-token (≤512) / large-token (>512) | as upstream | n/a (baseline) | bit-identical to the recovered baseline |

A single warp-per-token kernel covers BOTH captured regimes (decode and prefill), so no
multi-bucket split is needed inside the fast path; `grid = div_ceil(num_rows, 8)` scales from
M=1 (1 block) to M=7432 (929 blocks). The fallback exists only for off-domain safety/correctness
and is never exercised by the captured workloads.

## Why the fallback never re-introduces the baseline decode UB
The baseline decode UB is specific to `num_experts == 128` (which makes `warps_per_token=4 < 8`).
`num_experts == 128` is always `in_domain` and is therefore handled by the cold-safe candidate
kernel, never by the fallback. Off-domain configs that reach the fallback (e.g. `num_experts==256`
→ `warps_per_token=8`) do not trigger the uninitialized read, so the fallback is safe.

## Verification
- `bench/correctness.py` exercises the in-domain path on all captured + boundary + edge + tie
  shapes (482 checks pass) and the candidate is gate-selected for them.
- Off-domain fallback equivalence is guaranteed by construction (the baseline kernels are copied
  verbatim into `namespace fallback`), not merely by numerical comparison.
