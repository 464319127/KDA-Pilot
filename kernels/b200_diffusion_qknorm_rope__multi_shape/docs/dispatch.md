# Dispatcher decision table — b200_diffusion_qknorm_rope__multi_shape

`src/wrapper.py::fused_inplace_qknorm_rope` routes each call to one of two paths,
preserving the exact SGLang callsite contract and mutating Q/K in place.

## Routing gate (`_supported`)

Native CUDA fast path is taken iff ALL hold; otherwise → SGLang baseline:
- `is_neox == False`
- q, k CUDA, bf16, 3-D `[tokens, heads, 128]`, contiguous, equal token & head counts
- `head_dim == 128` and `rope_dim == 128`
- q_weight, k_weight bf16 contiguous, numel 128
- cos_sin_cache float32 contiguous, last dim 128
- positions 1-D int32 or int64, length == num_tokens

## Per-bucket decision (round 0, candidate v2, B200 GPU 0, idle)

| Bucket | Shapes (tokens) | Path | Baseline us | Cand us | Speedup | Promote? |
|---|---|---|---|---|---|---|
| large (image) | 4096 / 4128 / 7904 / 8424 | cuda | 30.9–79.6 | 32.9–86.2 | 0.91–0.94x | not yet (still behind; large shapes dominate model wall-clock) |
| tiny | 19 / 32 / 47 / 189 / 195 | cuda | 7.6–7.6 | 6.4–6.4 | 1.18–1.20x | yes-leaning (launch-bound win) |
| geomean | all 10 | — | — | — | all 1.049x / large 0.927x / tiny 1.187x | hold for v3 |

## Fallback bucket (correctness-only; not optimization shapes)

| Tuple | Path | Status |
|---|---|---|
| head_dim 64 / 256 | fallback | correct vs oracle |
| rope_dim < head_dim (e.g. 64) | fallback | correct vs oracle |
| is_neox == True | fallback | correct vs oracle |
| non-bf16 / non-contiguous / CPU / unequal Q-K heads | fallback | correct vs oracle |
| int32 positions @ production signature | cuda (native) | correct vs oracle |

## Promotion stance (round 0)

Hold. The candidate wins overall geomean (1.049x) and the launch-bound tiny
shapes, and is at near-parity kernel GPU time with the baseline on large shapes
(NCU: 60.1 vs 59.0 us), but is still ~6–9% behind on the large-token wall-clock,
which dominates real diffusion runs. A v3 (128-bit vectorization) and a
cold-cache roofline characterization are the gating work before promotion.
