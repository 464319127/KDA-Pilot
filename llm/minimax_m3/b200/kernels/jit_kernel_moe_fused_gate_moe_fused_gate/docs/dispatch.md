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

## Fallback safety scope (the fallback is bit-identical to the baseline, NOT a UB fix)
The fallback is a **verbatim copy** of the recovered baseline, so it reproduces baseline behavior
exactly — **including the latent decode UB**. The UB affects the small-token (`num_rows<=512`) path
whenever `div_ceil(num_experts,32) < 8` (i.e. `num_experts <= 224`, notably `num_experts==128`).

The candidate's cold-safe kernel covers **only the exact captured production config** (`in_domain`).
Therefore:
- **Captured MiniMax-M3 config (E=128 + the exact scalars):** always `in_domain` → cold-safe
  candidate kernel → UB removed. This is the only config in the captured workload set.
- **Off-domain `E=128` with non-captured scalars** (e.g. `topk!=5`, `scoring_func=1`, `rsf!=2.0`):
  fails the gate → fallback → baseline E=128 small-token → **same UB on a cold context**. The
  fallback does NOT make these safe; they are simply not part of this task's captured workloads.
- **Off-domain `E in {256,512}`:** `warps_per_token>=8` → the baseline small-token path is safe.

In short, the candidate **removes the decode UB for the captured config**; it does not (and is not
scoped to) fix the upstream baseline bug for arbitrary off-domain `E=128` configs, which inherit
baseline behavior including the UB. Fixing the upstream kernel generally is out of scope here.

## Verification
- `bench/correctness.py` exercises the in-domain path on all captured + boundary + edge + tie
  shapes (754 checks pass) and the candidate is gate-selected for them.
- Off-domain fallback equivalence is guaranteed by construction (the baseline kernels are copied
  verbatim into `namespace fallback`), not merely by numerical comparison.
