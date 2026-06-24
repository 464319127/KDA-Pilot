# Results — `sgl_kernel.topk_sigmoid` on B200

Host `ion-b200` (sm_100) · baseline = upstream SGLang `main` @ `5e6d7c1615a95dc5f98e69b4b18af0ae160b10b8` ·
candidate sha256 `08d131b8…` (`solution/topk_sigmoid_candidate.cuh`) · build `-std=c++17 -O3
-gencode=compute_100,sm_100`, no `--use_fast_math` · torch 2.11.0+cu130 / CUDA 13.0 / tvm_ffi 0.1.9.

## Verdict: GO

A single fused native-CUDA kernel replaces the baseline's two-launch workspace path (forced because
`num_experts = 288` is not a power of two) and is faster on **every** captured production row, while
remaining **bit-faithful** to the recovered baseline (exact top-8 expert ids; fp32 weights within
atol/rtol 1e-5). Promotion is judged at the strict op ABI (DEC-1).

- **Production geometric-mean speedup = 1.4381×** over 24 deduplicated captured rows (min 1.168×, max 2.637×).
- Active win mechanism: the candidate computes sigmoid + biased selection + top-8 + unbiased weights +
  renormalization in **one launch with shared-memory reductions and no workspace**, eliminating the
  baseline's `[N,288]` fp32 workspace round-trip + second kernel launch.

## Correctness (functional, GPU; 34/34 + 26/26 adversarial)

`bench/correctness.py` → **34/34 PASS**:
- All 24 production rows: candidate == recovered baseline EXACT on selected expert ids AND within fp32
  atol/rtol 1e-5 on weights; AND == independent fp32 torch oracle (sigmoid+bias selection, unbiased
  `gather`, renormalize) — matching upstream `test_topk_sigmoid_renormalize_correction_bias`.
- Route coverage proven: all 24 production rows take the fast path (`route==1`); 6 regression rows
  (non-contiguous, fp16, bf16, experts=64, topk=4, renormalize=False) route to the baseline
  (`route==0`) and stay correct — a silent fallback cannot masquerade as a candidate run.
- 4 constructed tie rows (identical logits, block-tied bias, equal score / different sigmoid, forced
  sigmoid tie) match the baseline exactly (lower-index tie-break).
- Output buffers poisoned (NaN/-17) each run; `gating_output` verified not mutated (read-only).

`bench/stress_adversarial.py` → **26/26 realistic configs** candidate == baseline (exact ids + fp32
weights): bias scales {0.1, 0.5, 1, 2, 3} × 5 seeds + extreme logits ×30. The only divergence is the
pathological `all-bias=-5` corner (all `sigmoid+bias` scores < -1), where the baseline's `-1.f`
sentinel in `moeTopK` degenerates; this is outside the realistic Step-3.7 correction-bias range and is
documented as a known boundary (the candidate computes the genuine top-k; both agree whenever ≥ topk
experts have score > -1, which holds for every realistic and tested input).

## Benchmark (idle B200 GPU 2; CUDA-event, inner-loop amplified, 7 trials, interleaved A/B)

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

**Headline: equal-weight geomean over the 24 production rows = 1.4381×.** Every production row is
faster. Fallback rows are ~1.0× (candidate == baseline), confirming the fallback is correct and free.

Call-weighted context (secondary, per DEC-1): N=1 decode dominates captured calls (5376 of 18816), so
the aggregate captured-latency saving is weighted toward the modest decode win; the large-prefill rows
contribute the largest per-call savings.

## Controlled launch-floor probe (empty kernel @ candidate grid, idle GPU 2)

| N | 1 | 16 | 80 | 1579 | 10207 | 16883 |
|---|---|----|----|------|-------|-------|
| floor median µs | 4.16 | 3.84 | 3.84 | 3.86 | 8.23 | 10.29 |

The ~3.8–4.2 µs small-N floor shows the candidate's ~12.34 µs decode latency is dominated by host-side
marshalling of 5 tensor arguments (paid equally by both sides through the identical ABI); the decode
win (1.168×) is the candidate removing the baseline's second launch + `torch::empty` workspace
allocation. At large N the floor (≤10.3 µs) is far below the candidate's tens-to-90 µs → real work.

## NCU bottleneck attribution (`--set basic`, N=16474, idle GPU 2)

| Kernel | Duration µs | Compute (SM) % | Memory % | DRAM % | Occupancy % |
|--------|-------------|----------------|----------|--------|-------------|
| baseline `moeSigmoid` | 33.79 | 46.2 | 12.8 | 8.45 | 66.7 |
| baseline `moeTopK` | **375.36** | **82.7** | 34.1 | 0.76 | 95.1 |
| candidate fused | **160.70** | 69.0 | 43.7 | 1.78 | 93.2 |

Roofline read: **no kernel is DRAM-bandwidth-bound** (DRAM ≤ 8.5%). The baseline is dominated by
`moeTopK` (375 µs, compute/reduction-bound at 82.7% SM — the 8 sequential `cub::BlockReduce` argmaxes
over the global workspace). The candidate replaces **both** baseline kernels with one (161 µs vs the
baseline's 409 µs total → ratio ~2.54×, matching the wall-clock large-N ratio) by doing the selection
in shared memory in a single pass. (NCU absolute durations are inflated by instrumentation/replay vs
wall-clock; ratios and bottleneck attribution are the valid signal.) Byte counts confirm the candidate
moves ~3× less global traffic at N=16474 (~20 MB vs ~58 MB: gating read + workspace write + workspace
read + outputs).

Warp-specialization profiling: **N/A** — the candidate is a straightforward block-reduction kernel with
no producer/consumer warp roles (no mbarrier / named-barrier / pipeline), so the
warp-specialization-report-skill does not apply; NCU + the floor probe are the evidence here.

## Independent Codex cross-check

An independent Codex (gpt-5.5:high) review **upheld the GO** at the op-ABI scope ("exact expert-id
agreement, fp32-close weights, tie-break tests, fallback route tests, every captured production row
faster; the large-N gain is explained by removing the repeated CUB reductions + extra launch/workspace
path, not a measurement artifact"). Its main correctness caveat — the baseline `-1.f` sentinel corner —
was closed by `bench/stress_adversarial.py` (26/26 realistic configs match; the degenerate
all-scores-<-1 boundary is documented above). Its scope caveat is honored: this result is "faster at
the strict op ABI without CUDA graph," **not** a full-server end-to-end claim.

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
