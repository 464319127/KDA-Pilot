# RESULTS_SM103 — mega_kernel task 01: mnnvl fused AR+add+rmsnorm, jit_kernel port + bs=1 specialization

Host: rx devbox `glm52-bs1-opt` (verda worker `light-face-hides-fin-03-1`),
8x NVIDIA B300 SXM6 AC, sm_103a, CUDA 13.0.88, NV18 full mesh, NVLS
multimem; GLM-5.2-FP8 bs=1 MTP (EAGLE 5-1-6) serving baseline 376.06 tok/s
official (campaign record: `config.toml` `serving_context` /
`mega_kernel/README.md`; pre-task official artifact copied to
`docs/serving_runs/baseline_official_pretask/`). Kernel: `oneshotAllreduceFusionKernel` (flashinfer 0.6.12),
157.3 calls/iter x 8.3-8.5 us = ~1.31 ms/iter (SHAPES.md re-verified on box
2026-07-10). Full method + deviations: `docs/benchmark_method.md`; complete
measurement narrative: `docs/run_log.md`; per-run records:
`bench/results.jsonl`.

## P0 — verbatim port into sglang jit_kernel (`solution/mnnvl_ar_fused/`)

| Gate | Result |
|---|---|
| (a) bf16 bit-exact vs flashinfer original, all 8 ranks, `out` + `residual_out`, T in {1,6} | **PASS** — on randn inputs AND an adversarial bf16 value zoo (subnormals, signed zeros, extremes), eager + graphs, pdl 0/1. fp32-fallback path not needed |
| (b) isolated perf within +-3% | **PASS** — port/original speedup 0.9739 (T=6), 0.9956 (T=1); noise floor 0.94-1.3% |
| (c) serving env gate `SGLANG_JIT_MNNVL_AR=1` (default OFF) | **PASS** — flag-OFF byte-inert (40/40 records identical, 376.53 tok/s); flag-ON 40/40 identical, 376.27 tok/s (>= 376); resident serving restored and re-verified afterward |
| Graph-replay stability (hardened detector: 3 rotating input banks, in-graph NaN poisoning, per-bank references) | **PASS** — 50,000 bit-checked rounds/row + 1000 plain replays, zero mismatches, pdl=1 |

The single load-bearing discovery ("verbatim source is not enough"): the
production baseline binary is the flashinfer-jit-cache AOT wheel, a
FAST-MATH build (SASS: 96,240 .FTZ modifiers, MUFU.RCP division). An IEEE
build of the byte-identical source flips output zero-signs on bf16-subnormal
gammas — invisible on randn data, deterministic 4/40 greedy-text divergence
in serving. Fix: `--use_fast_math` on the port (baseline-matching flag
symmetry), verified by a value-zoo A/B (0 mismatches) and the serving gate
(40/40). Diagnosis chain (route/module/scope bisections, SASS fingerprints):
`docs/run_log.md` "Flag-ON Divergence Diagnosis".

## P1 — bs=1 specialization (`mnnvl_ar_fused_opt.cuh`)

Byte-identity constraint: the promote gate requires greedy output identical
to stock, so candidates must preserve all value-bearing arithmetic, order,
and launch geometry (the norm's cross-warp fp32 tree is geometry-dependent).
Grid/block re-configs from the original idea pool are value-breaking under
this contract and were ruled out by analysis (recorded, not attempted).

Candidate (promoted): NumTokens/TokenDim template constants (index math
folds; the norm division keeps its RUNTIME operand so quotient bits stay
identical) + gamma/residual prefetch hoisted before the Lamport spin (load
latency hides behind the peer wait). Dispatch: frozen shapes -> constant
instantiations; everything else -> generic verbatim dispatch in the same
entry; serving keeps stock route for non-oneshot/prefill sizes and when the
env gate is off.

### Harness A/B (8x 50-round CUDA graphs, concurrent replay, wall/round, pdl=0)

Two campaigns (devbox released + reacquired mid-task; every gate re-verified
on the fresh box). The fresh-box campaign is the artifact-backed record
(raw samples in `bench/results.jsonl`):

| Campaign | Row | ported baseline us | specialized us | speedup | noise spread |
|---|---|---:|---:|---:|---:|
| pre-release box | T=6 | 6.831 | 6.726 | 1.0157 | 0.94% |
| pre-release box | T=1 | 6.576 | 6.435 | 1.0219 | 0.14% |
| **fresh box (jsonl)** | T=6 | 6.858 | 6.773 | **1.0126** | 0.36% |
| **fresh box (jsonl)** | T=1 | 6.556 | 6.409 | **1.0230** | 0.46% |

Headline geomean **1.0188 / 1.0178** (pre-release / fresh-box; 25 trials,
interleaved A/B, fresh inputs/trial, pre-timing bit-exact check enforced).
Hard promotion bar (geomean > 1.0 beyond noise, no production row < 0.97x):
**MET in both campaigns**. Candidate is bf16 bit-exact vs the flashinfer
original (randn + value zoo, both boxes) and 50,000-round stability-clean
(pdl=1, both boxes).

### NCU breakdown (solo pre-fed launch, T=6, application replay)

Durable text exports: `docs/profiler_evidence/ncu_{jit,opt}_t6_details.txt`
(fresh-box re-run; raw `.ncu-rep` stays on-box per artifact hygiene — the
repo gitignores `**/*.ncu-rep`). Both campaigns agree:

| Metric | ported baseline | specialized |
|---|---:|---:|
| Duration under NCU (pre-release / fresh box) | 16.29 / 16.93 us | 15.36 / 15.36 us |
| Elapsed cycles (pre-release / fresh box) | 17,777 / 18,466 | 16,746 / 16,756 |
| SM / DRAM throughput (fresh box) | 0.90% / 0.60% | 0.94% / 0.66% |
| Registers per thread | 61 | 58 |
| Achieved occupancy / no-eligible (fresh box) | 9.33% / 91.66% | 9.21% / 91.00% |
| Geometry | grid 24 x block 192 (6 tokens x 4-block clusters) | identical (pinned for byte-identity) |

Reading: the kernel is pure latency — warps park on the NVLS fan-out +
Lamport arrival wait at ~0.9% SM utilization. The solo delta is consistent
with local tail trimming (fewer instructions, spin-hidden operand loads);
the production-relevant readout is the in-serving per-call profile below.
Named bound on further value-preserving gains: peer-arrival latency
dominates and is protocol-level, not compute-level.

### FINAL promoted kernel (round 1 adoption): block-arrival specialization

After the round-0 promote, the fence/flag-granularity route evaluation
produced a byte-preserving improvement (block-granular arrival, see the
route-closure section) which passed the FULL ladder and a FRESH promote
pass, superseding the cluster-arrival candidate as the shipped kernel:

| Gate (adopted `oneshotArFusedNormConstKernelBA`) | Result |
|---|---|
| bit-exact vs flashinfer original (randn + value zoo, pdl 0/1) | PASS (both rows, all ranks, both outputs) |
| hardened stability, pdl=1, 1000 replays | PASS (50,000 rounds/row, zero mismatches) |
| harness A/B vs ported baseline (25 trials, same-session pairing) | T=6 **1.0563** (6.902 -> 6.534us), T=1 **1.0884** (6.535 -> 6.004us), geomean **1.0722** — far beyond the 0.36-0.46% noise floor |
| serving warm sanity / sanity-2 | **381.54 / 380.52 tok/s** (>= 378), 40/40 identical |
| OFFICIAL 3x40 | **381.42 tok/s** (baseline official 376.06 -> **+1.4%**), all runs 40/40 byte-identical |
| in-serving per-call | **7.6 us** (13,600 calls) vs 8.3-8.5 baseline capture (-8..-11%) |
| serving restored (flag unset) | 378.14 tok/s, 40/40 identical |

<=7.0us target status: **still MISSED at 7.6 us**, but the gap to the
serving-convention target shrank from ~1.3-1.5 us to ~0.6 us. Named bound
unchanged: NVLS peer-arrival latency; the remaining byte-identity-safe
levers are exhausted (see route closure).

### Round-0 promote record (cluster-arrival candidate, superseded)

Executed on the reacquired box (same worker, rebuilt env; all P0/P1 gates
re-verified there first — cross-box reproduction: 25-trial geomean 1.0178 vs
1.0188 pre-release; fi-vs-opt value zoo 0 mismatches; stability clean; new
base text record captured on the same box before comparison):

| Promote step | Result |
|---|---|
| warm-up sanity 1x40 (ON+OPT) | 376.94 tok/s, 40/40 records identical to base |
| sanity 2 (warmed) | **378.86 tok/s (>= 378)**, 40/40 identical — promote bar MET |
| OFFICIAL 3x40 | **379.2 tok/s overall; every run 40/40 byte-identical** (baseline official: 376.06) |
| in-serving per-call (bounded profile, fresh box) | `oneshotArFusedNormConstKernel` live: 13,440 calls, mean **8.2 us** (pre-release box read 8.0 us) vs baseline capture 8.3-8.5 us |
| serving restored (flag unset) | 375.5 tok/s, 40/40 records identical to base |

**<=7.0us target: MISSED** (8.0-8.2 us in-serving per-call; stretch 5 us not
approached). Named active bound: the kernel is latency-bound on the NVLS
fan-out + Lamport peer-arrival wait (NCU: 91% no-eligible at 0.9% SM
throughput); under the byte-identical-output promote contract the
value-preserving optimization space (constant folding, spin-hidden
prefetch) is exhausted at ~2% wall/round // ~0.1-0.3 us per-call. Reaching
7 us would require value-affecting changes (launch-geometry/reduction-tree
restructuring or protocol changes), which this task's promote gate forbids;
per the standing promotion policy the numeric target is direction, and the
hard bar (geomean > 1.0 beyond noise, no row < 0.97x, sanity >= 378 +
byte-identical + official 3x40) is fully met.

## P1 route evaluation closure (full idea-pool disposition)

See `docs/results.md` "P1 route evaluation closure" for the evidence table
and `docs/dispatch.md` for the resulting dispatch. Summary:

- Constant-fold + spin-hidden prefetch: ADOPTED (the P1 candidate).
- PDL entry: EVALUATED with a predecessor-aware probe (`--mode pdlprobe`,
  jsonl-backed records in `bench/results.jsonl`) — pdl1-vs-pdl0 deltas
  +0.03%..+0.21%, inside noise, bit-ok; no dispatch change; kernel PDL hooks
  preserved verbatim.
- Fence/flag granularity: byte-preserving BLOCK-ARRIVAL transform found
  (drops ctaArrive's cluster.sync; rotation counts blocks), bit-exact incl.
  value zoo, stability-clean, +6.9% @ T=1 / +0.8% @ T=6 over the
  cluster-arrival candidate — ADOPTED into the specialized entry behind the
  full ladder + a fresh promote pass (numbers below).
- Epilogue fold: NO-GO for this config — trace adjacency
  (`docs/profiler_evidence/trace_adjacency_task01_ba.txt`) shows the AR
  feeds closed-source cuBLAS `nvjet` GEMMs directly (bf16-dense path has no
  quant kernel after the AR); folding into cubins is outside the campaign
  rule; conditional follow-up only for an fp8-quant-after-AR config.

## Failed / ruled-out routes

- Unicast-push transport rebuild: pre-judged dead by prior art (36.6 us,
  ~25 us world-independent protocol cost; `common/prior_art/old02_ar_norm_RESULTS.md`) — not revisited.
- IEEE (default-flag) build of the verbatim source: VALUE-DIVERGENT vs the
  deployed fast-math baseline on subnormal/zero-sign classes (serving texts
  4/40) — replaced by baseline-matching `--use_fast_math`.
- Grid/block re-configuration (e.g. single 768-thread block per token, no
  CGA) and any norm-reduction restructuring: value-breaking under the
  byte-identical promote contract (fp32 tree order changes) — ruled out by
  analysis for this task's promote path; a legitimate future direction only
  if the byte-identity requirement is relaxed by the owner.
- NCU application-replay on the CONCURRENT 8-rank harness: hangs by design
  (per-launch serialization starves the collective's peers) — replaced by
  the pre-fed solo-rank mode (`bench/ar_harness.py --mode ncusolo`).
- PDL in back-to-back and predecessor-adjacent settings: no measurable win
  at these payloads (probe data above) — not a dispatch lever here.
- Epilogue folding into the bf16-dense consumer chain: no-go (closed-source
  consumers; no post-AR quant kernel in this config).

## Warp-specialization profiling applicability

Not applicable: neither the port nor the specialized candidate uses
producer/consumer warp roles (uniform per-thread protocol; no mbarrier/named
barrier pipelines). Recorded per the task contract.
