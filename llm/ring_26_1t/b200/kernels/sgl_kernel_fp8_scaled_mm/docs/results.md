# Results — `sgl_kernel.fp8_scaled_mm` on B200 (Ring-2.6-1T)

## Verdict: PROMOTE (decode M=1 regime), fall back elsewhere

A native-CUDA FP8 GEMV wins the **M=1 decode regime** — the single largest
captured shape (M=1 = 38,427 calls) — with **no regression on any covered
shape**, clearing the promotion bar (per-regime geomean ≥ 1.10, statistically
significant). All other regimes (small-M, prefill, and the largest-work M=1
shapes) fall back to the recovered CUTLASS baseline at parity. Per the resolved
success policy (per-regime win + fallback), this is a promotion.

Method/provenance: `docs/baseline_source.md` (baseline lineage + sm100 dispatch),
`docs/run_log.md` (host/GPU/idle/commands/versions), `docs/dispatch.md` (routing).
All runs on `ion-b200` GPU id 3 (B200 sm_100, CUDA 13.0, torch 2.11+cu130,
CUTLASS@57e3cfb4), GPU verified idle before+after. Timing: standard harness
(`standalone_llm_benchmark_template.py`), CUDA events, inner-loop amplification,
isolated subprocess, 7 trials; speedup = baseline_median/candidate_median.
Correctness: 291/291 (286 production + 5 edge) vs fp32-dequant oracle AND baseline.

## Per-regime results

### M=1 decode — PROMOTED (covered shapes, candidate = FP8 GEMV)

| shape (M·K·N) | baseline µs | candidate µs | speedup |
|---|---|---|---|
| 1·1536·1536 | 12.22 | 4.12 | **2.97×** |
| 1·256·8192 | 12.24 | 4.68 | **2.61×** |
| 1·1024·8192 | 12.18 | 6.17 | **1.97×** |
| 1·8192·512 | 12.35 | 8.24 | **1.50×** |
| 1·8192·1024 | 12.19 | 8.24 | **1.48×** |
| 1·2304·8192 | 12.29 | 10.29 | **1.20×** |
| 1·8192·2112 | 12.36 | 10.85 | **1.14×** |

Covered-shape geometric mean ≈ **1.73×**; every covered shape ≥ 1.10; no covered
shape regresses.

### M=1 fallback (excluded: K≥4096 AND N≥3072)

| shape | baseline µs | candidate µs | speedup |
|---|---|---|---|
| 1·8192·3072 | 12.21 | 12.25 | 1.00× (fallback) |
| 1·8192·4608 | 12.23 | 12.21 | 1.00× (fallback) |

(GEMV measured 0.86× / 0.73× here when forced — fp8-decode-instruction-bound;
the dispatch routes them to the baseline so there is no regression.)

### Small-M (M=23,32,57) and prefill (M≥684) — fallback at parity

Representative measured (candidate routes to baseline, route==0): m23·1024·8192
1.00×, m32·8192·512 ~1.0×, m57·1024·8192 ~0.98× (noise), m2923·1024·8192 1.00×,
m12653·8192·512 1.00×, m27503·512·2048 1.00×. No specialized path yet; identical
to baseline by construction (within trial noise).

## Roofline / active-bound analysis (NCU on GPU 3)

**Baseline, M=1 (structural waste).** The recovered sm100 path uses a CUTLASS MMA
tile with CTA tile-M = 64 even for M=1 (Gemm16), so ~63/64 of the tensor-core M
dimension is unused. Wall-clock for 1·1024·8192 (12.2 µs over ~8.4 MB of B
traffic) ⇒ ≈ 690 GB/s ≈ **8.6% of the B200's ~8 TB/s HBM** — far from
bandwidth-efficient in decode.

**Candidate FP8 GEMV, M=1 (the win, and its ceiling).** NCU on 1·1024·8192
(covered): `dram__bytes_read` = **9.48% of peak**, `sm__throughput` = 37%,
achieved occupancy = 63%, dominant issue stall = **`long_scoreboard` (4.83)** ≫
`math_pipe_throttle` (2.53). So the GEMV is **memory-latency-bound** (waiting on
global loads), not bandwidth-saturated and not compute-bound. It beats the
baseline by avoiding the 64-row MMA waste and reading B once with fully-coalesced
`uint4` loads, but its own bound is load latency / insufficient memory-level
parallelism — which is exactly why it loses on the largest-work shapes (more
dependent fp8-decode work per column) and falls back there.

Named active bounds: baseline M=1 = tensor-core under-utilization (64-row tile) →
HBM far below peak; candidate M=1 = global-load latency (`long_scoreboard`),
headroom remains below the HBM-bandwidth roofline.

## Follow-ups (not blocking the promotion; bounded by the plan's upper path boundary)

1. **Higher-MLP GEMV** to lift the latency-bound ceiling (more in-flight loads:
   multiple columns/warp, K-split warps, or double-buffered prefetch) and a
   faster vectorized fp8→half decode — would widen coverage to the excluded
   large-K×N M=1 shapes.
2. **Swap-AB skinny tensor-core GEMM** for the hot small-M shapes (M=23,32,57; per
   KernelWiki `pr-vllm-27284`) — the path to extend the win past M=1. Currently
   these fall back at parity.
