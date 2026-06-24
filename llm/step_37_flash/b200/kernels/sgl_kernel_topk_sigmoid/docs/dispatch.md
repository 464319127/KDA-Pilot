# Dispatch — `sgl_kernel.topk_sigmoid`

A single fused candidate kernel covers the entire captured production regime, so there is one
fast-path bucket plus a baseline fallback. Dispatch is **host-only** (metadata checks on the
TensorView shapes/dtypes/strides/scalars — no device-memory reads, no host sync on the hot path),
implemented by `candidate_eligible(...)` in `solution/topk_sigmoid_ext.cu` and exposed for proof by
`topk_sigmoid_candidate_route(...)`.

## Buckets

| Bucket | Condition (all must hold) | Path | Launch geometry |
|--------|---------------------------|------|------------------|
| Fast path | gating `[N,288]` fp32 contiguous CUDA; topk_weights `[N,8]` fp32; topk_indices `[N,8]` i32; correction_bias `[288]` fp32; renormalize=True | `tsc::topk_sigmoid_fused_kernel<288,8,128>` | grid = N blocks, block = 128 threads, 1 launch, no workspace |
| Fallback | anything else (num_experts≠288, topk≠8, dtype≠fp32, non-contiguous, renormalize=False, …) | recovered baseline `topk_sigmoid(...)` | upstream launcher (workspace + 2 launches for non-pow2 experts) |

The 24 deduplicated captured production rows (N from 1 to 16883) all satisfy the fast-path
condition → `route==1`, asserted per row in `bench/correctness.py`. The 6 regression rows
(non-contiguous, fp16, bf16, experts=64, topk=4, renormalize=False) all take the fallback →
`route==0`, also asserted. This proves the fast path genuinely covers production and a silent
fallback cannot masquerade as a candidate run.

## Per-bucket measured behavior (idle B200 GPU 2)

| Regime | N | baseline median µs | candidate median µs | speedup |
|--------|---|--------------------|---------------------|---------|
| decode | 1 | 14.42 | 12.34 | 1.168× |
| mid | 7–80 | 14.4–15.5 | ~12.34 (flat) | 1.17–1.25× |
| large prefill | 1579 | 30.98 | 14.39 | 2.153× |
| large prefill | 9030 | 131.06 | 51.40 | 2.550× |
| large prefill | 10207 | 146.90 | 57.99 | 2.533× |
| large prefill | 16206 | 224.99 | 86.65 | 2.596× |
| large prefill | 16474 | 229.36 | 86.97 | 2.637× |
| large prefill | 16883 | 233.40 | 91.06 | 2.563× |
| fallback | 64 | (baseline) | ~baseline (0.98–1.00×) | route==0 |

## Why one kernel suffices (no further specialization needed)

The fast path is flat at ~12.34 µs for N ≤ 80 (host/launch-bound; floor probe ~4 µs) and scales with
N beyond that. A single grid-stride-of-rows kernel already beats the baseline across the full N
range (1.168×–2.637×), so per-shape specialization is not required to win. Remaining candidate
headroom (it is SM/`__syncthreads`-latency-bound, not DRAM-bound — see `docs/results.md`) would come
from a warp-per-row shuffle reduction, recorded as out-of-scope follow-up, not a second dispatch
bucket.
