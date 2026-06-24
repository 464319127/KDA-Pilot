# Results — `sgl_kernel.topk_sigmoid` on B200

Host `ion-b200` (sm_100) · baseline = upstream SGLang `main` @ `5e6d7c1615a95dc5f98e69b4b18af0ae160b10b8` ·
candidate sha256 `5d1a6d24…` (`solution/topk_sigmoid_candidate.cuh`, Round-1 sentinel-fixed) · build `-std=c++17 -O3
-gencode=compute_100,sm_100`, no `--use_fast_math` · torch 2.11.0+cu130 / CUDA 13.0 / tvm_ffi 0.1.9.

## Verdict: GO

A single fused native-CUDA kernel replaces the baseline's two-launch workspace path (forced because
`num_experts = 288` is not a power of two) and is faster on **every** captured production row, while
remaining **bit-faithful** to the recovered baseline (exact top-8 expert ids; fp32 weights within
atol/rtol 1e-5). Promotion is judged at the strict op ABI (DEC-1).

- **Production geometric-mean speedup = 1.4754×** over 24 deduplicated captured rows (min 1.176×, max
  2.638×) on idle GPU 3 (Round 1 rerun after the sentinel fix; consistent with the Round 0 GPU-2 run
  at 1.4381× — large-N identical, small-N decode within host-overhead measurement noise).
- Active win mechanism: the candidate computes sigmoid + biased selection + top-8 + unbiased weights +
  renormalization in **one launch with shared-memory reductions and no workspace**, eliminating the
  baseline's `[N,288]` fp32 workspace round-trip + second kernel launch.
- The candidate now **replicates upstream `moeTopK`'s `-1.f`/index-0 argmax sentinel exactly**, so it
  is bit-faithful to the baseline for every input the host-only route predicate admits (any fp32
  `[N,288]` bias), not only a "realistic" bias range (Round 1 fix; see Correctness).

## Correctness (functional, GPU; 36/36 + 27/27 adversarial)

`bench/correctness.py` → **36/36 PASS**:
- All 24 production rows: candidate == recovered baseline EXACT on selected expert ids AND within fp32
  atol/rtol 1e-5 on weights; AND == independent fp32 torch oracle (sigmoid+bias selection, unbiased
  `gather`, renormalize) — matching upstream `test_topk_sigmoid_renormalize_correction_bias`.
- Route coverage proven: all 24 production rows take the fast path (`route==1`); 6 regression rows
  (non-contiguous, fp16, bf16, experts=64, topk=4, renormalize=False) route to the baseline
  (`route==0`) and stay correct — a silent fallback cannot masquerade as a candidate run.
- **Missing-bias fallback (AC-4):** `fb_missing_bias` uses the no-bias ABI variants; `route_nobias==0`
  (no fast path without bias), candidate == baseline == a no-bias sigmoid-topk oracle, poison/in-place verified.
- **Fallback dtype safety (AC-4):** `fp16_bias_fallback` — an fp16 `correction_bias` is off-domain
  (`route==0`); the local ABI bridge preserves the real dtype, so the vendored baseline's
  `correction_bias must be float32` check raises cleanly on both sides (no out-of-bounds reinterpretation).
- 4 constructed tie rows (identical logits, block-tied bias, equal score / different sigmoid, forced
  sigmoid tie) match the baseline exactly (lower-index tie-break).
- Output buffers poisoned (NaN/-17) each run; `gating_output` verified not mutated (read-only).

`bench/stress_adversarial.py` → **27/27 PASS** candidate == baseline (exact ids + fp32 weights): bias
scales {0.1, 0.5, 1, 2, 3} × 5 seeds + extreme logits ×30 **plus the `all_neg_bias_-5` sentinel corner**
(all `sigmoid+bias` scores < -1). The candidate now **replicates upstream `moeTopK`'s `-1.f`/index-0
sentinel exactly**, so it matches the baseline bit-for-bit on that corner too (no "realistic bias"
carve-out). This matters because the host-only route predicate cannot inspect bias values, so any fp32
`[N,288]` bias — including one below -1 — is inside the fast-path domain and must be faithful.

## Benchmark (idle B200; CUDA-event, inner-loop amplified, 7 trials, interleaved A/B)

Per-row median µs below are the Round 0 detail run on GPU 2; the Round 1 rerun after the sentinel fix
(idle GPU 3) gave geomean **1.4754×** (decode 1.252×, large-prefill stable within noise).

| Regime | N | baseline median µs | candidate median µs | speedup |
|--------|---|--------------------|---------------------|---------|
| decode | 1 | 14.42 | 12.34 | 1.168× |
| mid | 7–80 | 14.4–15.5 | ~12.34 (flat) | 1.17–1.25× |
| large prefill | 1579 | 30.98 | 14.39 | 2.153× |
| large prefill | 9030 | 131.06 | 51.40 | 2.550× |
| large prefill | 10207 | 146.90 | 57.99 | 2.533× |
| large prefill | 16206 | 224.99 | 86.65 | 2.596× |
| large prefill | 16474 | 229.36 | 86.97 | **2.637×** |
| large prefill | 16883 | 233.40 | 91.06 | 2.563× |
| fallback (6 rows) | 64 | — | ≈ baseline | 0.98–1.00× (route==0) |

**Headline: equal-weight geomean over the 24 production rows = 1.4754× (Round 1, idle GPU 3; Round 0
on GPU 2 gave 1.4381×).** Every production row is faster in both runs (min 1.176×); large-N is stable
across runs, small-N decode varies within host-overhead noise (1.168× GPU-2 → 1.252× GPU-3). Fallback
rows are ~1.0× (candidate == baseline), confirming the fallback is correct and free.

Call-weighted context (secondary, per DEC-1): N=1 decode dominates captured calls (5376 of 18816), so
the aggregate captured-latency saving is weighted toward the modest decode win; the large-prefill rows
contribute the largest per-call savings.

## Controlled launch-floor probe (empty kernel @ candidate grid, idle GPU 3, Round-1 build)

Rerun on the Round-1 (sentinel-fixed) build, idle GPU 3:

| N | 1 | 16 | 80 | 1579 | 10207 | 16883 |
|---|---|----|----|------|-------|-------|
| floor median µs | 9.36 | 8.50 | 8.88 | 8.12 | 8.23 | 10.31 |

The small-N floor (~8–9 µs here on GPU 3; the Round-0 GPU-2 run measured ~4 µs — a per-GPU/clock-state
difference, not a kernel change) shows the candidate's ~12 µs decode latency is dominated by host-side
marshalling of 5 tensor arguments (paid equally by both sides through the identical ABI); the decode
win is the candidate removing the baseline's second launch + `torch::empty` workspace allocation. At
large N the floor (≤10.3 µs) is far below the candidate's tens-to-90 µs → real work. (Floor probes the
empty kernel, unaffected by the sentinel fix; the rerun confirms the small-N-launch-bound reading.)

## NCU bottleneck attribution (`--set basic`, N=16474, idle GPU 3, Round-1 build)

Rerun on the Round-1 (sentinel-fixed) build — confirms the Round-0 attribution (numbers within noise):

| Kernel | Duration µs | Compute (SM) % | Memory % | DRAM % | Occupancy % |
|--------|-------------|----------------|----------|--------|-------------|
| baseline `moeSigmoid` | 33.44 | 45.2 | 12.6 | 8.54 | 66.0 |
| baseline `moeTopK` | **365.41** | **82.6** | 34.1 | 0.78 | 95.1 |
| candidate fused | **154.88** | 69.7 | 43.7 | 1.84 | 93.1 |

Roofline read: **no kernel is DRAM-bandwidth-bound** (DRAM ≤ 8.5%). The baseline is dominated by
`moeTopK` (365 µs, compute/reduction-bound at 82.6% SM — the 8 sequential `cub::BlockReduce` argmaxes
over the global workspace). The candidate replaces **both** baseline kernels with one (155 µs vs the
baseline's 399 µs total → ratio ~2.58×, matching the wall-clock large-N ratio) by doing the selection
in shared memory in a single pass. The sentinel fix (an argmax init constant) left the bottleneck
attribution unchanged vs Round 0 (candidate 154.88 vs 160.70 µs, moeTopK 365 vs 375 µs — within
run-to-run noise). (NCU absolute durations are inflated by instrumentation/replay vs wall-clock; ratios
and bottleneck attribution are the valid signal.) Byte counts confirm the candidate moves ~3× less
global traffic at N=16474 (~20 MB vs ~58 MB: gating read + workspace write + workspace read + outputs).

Warp-specialization profiling: **N/A** — the candidate is a straightforward block-reduction kernel with
no producer/consumer warp roles (no mbarrier / named-barrier / pipeline), so the
warp-specialization-report-skill does not apply; NCU + the floor probe are the evidence here.

## Independent Codex review

Two independent Codex passes (gpt-5.5:high) were run. (1) A manual cross-check upheld the GO at op-ABI
scope. (2) The formal RLCR Round 0 review was more rigorous and returned **NOT COMPLETE**, flagging
three gaps: the `-1.f` sentinel divergence was an **in-contract correctness bug** (the host-only route
predicate cannot exclude a `bias < -1` vector, so the degenerate corner is inside the fast-path
domain); the AC-1 runtime probe was not recorded; and the AC-4 missing-bias fallback was omitted.

**Round 1 closed all three:** the candidate now replicates upstream `moeTopK`'s `-1.f`/index-0 sentinel
exactly (adversarial stress 27/27, including the corner — no carve-out); the runtime probe is recorded
(`docs/baseline_source.md`); and a no-bias ABI fallback path + `fb_missing_bias` test was added
(correctness 35/35). The scope caveat is honored: this is "faster at the strict op ABI without CUDA
graph," **not** a full-server end-to-end claim.

## Remaining headroom (out of scope for this op-ABI GO)

The candidate is SM/`__syncthreads`-latency-bound (69% SM, DRAM 1.8%), not bandwidth-saturated — a
warp-per-row shuffle reduction (like the upstream power-of-two `topkGatingSigmoid` fast path) could cut
the 8 sequential block-reduction barriers and push higher. Larger serving wins would also require
fusing the router with downstream MoE dispatch or CUDA-graph capture of the decode path. These are
documented follow-ups, not part of this strict-op-ABI promotion.

## Provenance

- Baseline: `docs/baseline_source.md` · Method/flags/timing: `docs/benchmark_method.md` ·
  Dispatch/route table: `docs/dispatch.md` · Host/GPU/idle/commands: `docs/run_log.md`.
- Raw evidence kept local/remote and excluded from the PR: `bench/.build/`, `bench/results.jsonl`,
  `/tmp/topk_ncu.ncu-rep`, run logs.
