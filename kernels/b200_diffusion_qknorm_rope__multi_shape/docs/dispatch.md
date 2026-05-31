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

## Per-bucket decision (round 0, candidate v3 = 2-heads-per-warp + 128-bit, B200 GPU 0, idle)

Numbers below are the final benchmark: one kernel invocation per CUDA-event
sample, with a pristine Q/K reset before each timed sample; GPU idle
before=0%/4MB and after-idle (settled to 0% util) recorded in benchmark.csv.

| Bucket | Shapes (tokens) | Path | Baseline us | Cand us | Speedup | Promote? |
|---|---|---|---|---|---|---|
| large (image) | 4096 / 4128 / 7904 / 8424 | cuda | 46.6–96.1 | 42.4–79.3 | 1.10–1.21× | **yes** (wins the wall-clock-dominant bucket) |
| tiny | 19 / 32 / 47 / 189 / 195 | cuda | ~25.6–26.0 | ~23.5–23.8 | 1.08–1.11× | **yes** (launch/dispatch-bound; native path lighter than tvm-ffi) |
| geomean | all 10 | — | — | — | **all 1.111× / large 1.133× / tiny 1.091×** | **PROMOTE** |

Tiny-bucket absolute latencies include the per-sample reset+`synchronize()`
overhead (paid equally by both impls), so the *speedup ratio* is the meaningful
metric there. Single-call timing has ~±0.02 run-to-run variance on the geomeans;
the candidate wins every shape across runs.

Single CUDA path (v3) for the whole production signature; no per-bucket kernel
split is needed — the 2-heads-per-warp/128-bit kernel wins both buckets, so the
tiny-shape policy is "use the same native path" (it also beats the baseline there).

## Fallback bucket (correctness-only; not optimization shapes)

| Tuple | Path | Status |
|---|---|---|
| head_dim 64 / 256 | fallback | correct vs oracle |
| rope_dim < head_dim (e.g. 64) | fallback | correct vs oracle |
| is_neox == True | fallback | correct vs oracle |
| non-bf16 / non-contiguous / CPU / unequal Q-K heads | fallback | correct vs oracle |
| int32 positions @ production signature | cuda (native) | correct vs oracle |

## Promotion stance (round 0): PROMOTE

Candidate v3 beats the SGLang baseline on **every** configured shape — geomean
**1.111×** (large **1.133×**, tiny **1.091×**) — with correctness intact (21/21)
and a roofline/NCU explanation (latency-bound kernel; 128-bit + 2-heads-per-warp
lifted cold achieved DRAM BW ~30%→~40% of peak, NCU duration 60→50 µs). The
candidate also removed an uncoalesced cos/sin gather the baseline still carries.
Promoted via `scripts/export_kda_kernels/export.py b200_diffusion_qknorm_rope__multi_shape`
(EXPORTS = `{"fused_inplace_qknorm_rope": ...}`), with install/uninstall/status
and a post-install recursion-safe smoke verified.
