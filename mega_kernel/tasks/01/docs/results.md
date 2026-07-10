# Results — mega_kernel task 01 (mnnvl_ar_jit_bs1)

Benchmark convention: `bench/ar_harness.py` — 8x per-device 50-round CUDA
graphs, concurrent replay, wall/round, pdl=0, per config.toml; workspace =
single-process NVLS (`bench/sp_nvls_workspace.py`); rows = `bench/workloads.json`
(frozen). Provenance per record in `bench/results.jsonl`; method + deviations
D1-D4 in `docs/benchmark_method.md`. Serving-side reference (provenance, not a
gate): 8.3-8.5 us in-graph mean at 157.3 calls/iter (SHAPES.md re-verification).

## Immutable Baseline (flashinfer 0.6.12 original) — FROZEN 2026-07-10

15 trials, reps=10, rounds=50, fresh inputs per trial, resident serving
request-idle (0% util, memory-resident; noise floor quantified below).
Frozen copy: `bench/results_baseline_frozen.jsonl`. These numbers are never
edited; later runs are compared against re-measured baselines under identical
conditions, with this table as the anchor.

| Row | median us | mean | std | min | n |
|---|---:|---:|---:|---:|---:|
| T=6 (73.7 KB payload) | 6.917 | 6.961 | 0.339 | 6.433 | 15 |
| T=1 (12.3 KB payload) | 6.539 | 6.547 | 0.046 | 6.484 | 15 |

## Noise Floor (same-impl A/B, fi vs fi, 7 trials)

| Row | A median | B median | pseudo-speedup | spread |
|---|---:|---:|---:|---:|
| T=6 | 6.713 | 6.801 | 0.9871 | 1.29% |
| T=1 | 6.532 | 6.579 | 0.9928 | 0.72% |

"Beyond noise" for AC judgments = outside ~1.3% on median pseudo-speedup.
If any gate decision lands inside this margin, the measurement is repeated
with the resident serving stopped (documented restart) before concluding.

## Oracle floor measurement (correctness context)

flashinfer ORIGINAL vs composed fp32 oracle: max-rel(out) 3.904e-3 (T=6, 1
round) = one bf16 ulp (2^-8); 4.741e-3 after 3 rounds; T=1 3.942e-3. The
prompt's fallback figure "rel<1e-3" is unsatisfiable under max-rel for any
bf16 kernel (the baseline itself included); oracle gate runs at the contract
bf16 tolerance (atol 7e-2 / rtol 2e-2). Primary gate remains bitwise A/B.

## P0 Port — the "verbatim source is not enough" discovery (2026-07-10)

The production baseline binary is the AOT wheel `flashinfer-jit-cache
0.6.12+cu130` (flashinfer's loader prefers it; no local JIT build exists).
That wheel is a FAST-MATH build (SASS: 96,240 `.FTZ` modifiers, `MUFU.RCP`
approximate division), while a default TVM-FFI build of the very same
verbatim header is IEEE. The two differ bf16-bitwise ONLY on value classes
that `randn` inputs never produce — bf16-subnormal gammas (the wheel's
`FADD.FTZ(+0, gamma_subnormal)` flushes to zero before the multiply; the
IEEE build keeps the tiny value whose product casts to the opposite zero
sign) plus a rarer approximate-division channel. Real serving activations
contain those classes: the first flag-ON serving pass produced 4/40
base-identical greedy texts, fully deterministically (bisected via
route+stock-module = 40/40, SCOPE=attn = 4/40, run1==run2 = 40/40, value-zoo
A/B = zero-sign mismatches on `out` only, symbol signatures identical).

Resolution: the port keeps the source verbatim and matches the DEPLOYED
binary's float codegen with `--use_fast_math` (baseline-matching, not a
one-sided advantage — the baseline carries the same semantics). After the
rebuild: `bench/value_zoo_ab.py` (bf16 subnormals, signed zeros, extremes;
T in {1,6} x pdl in {0,1} x 3 seeds) = 0 mismatched elements.

## P0 Port — bit-exactness — PASS (fast-math build, 2026-07-10)

`bench/ar_harness.py --mode correctness --impls fi,jit`: bf16 BIT-EXACT vs
the flashinfer AOT original on all 8 ranks, both outputs, T=1 and T=6, on
randn inputs AND on the adversarial value zoo; also bit-exact at pdl=1.
Both sides show identical oracle max-rel (4.741e-3 @ T=6, 3.942e-3 @ T=1),
within contract tolerance. The AC-4.1 fp32-fallback path was NOT needed.

Port diff surface: `solution/mnnvl_ar_fused/csrc/mnnvl_ar_fused.cuh` differs
from `baseline/trtllm_mnnvl_allreduce.cuh` ONLY in the include block (3
flashinfer-internal includes -> `mnnvl_ar_fused_compat.cuh`); the binding
mirrors flashinfer's csrc binding with the include swapped. Build: TVM-FFI,
arch 10.3a, `-std=c++20 -O3 --expt-relaxed-constexpr --use_fast_math`
(matching the AOT baseline; see discovery above).

## P0 Port — isolated parity (+-3% gate) — PASS (fast-math build, 2026-07-10)

15 trials, reps=10, rounds=50, pdl=0, interleaved A/B, fresh inputs/trial,
pre-timing bit-exact check enforced per row:

| Row | fi median us | jit median us | speedup (fi/jit) | verdict |
|---|---:|---:|---:|---|
| T=6 | 6.738 | 6.918 | 0.9739 | within 3% (2.61% slower; noise floor 1.3%, T=6 trial std ~5%) |
| T=1 | 6.560 | 6.589 | 0.9956 | within 3% (0.44% slower, inside noise) |

Headline geomean (production rows) 0.9847. Earlier IEEE-build parity read
0.9916/0.9946 — the deltas sit inside trial noise; both builds pass the
gate. P1 optimization targets far beyond this margin regardless.

## P0 Port — graph-replay stability

First-cut early check (500 replays, 25,000 rounds/row, zero mismatches) was
DOWNGRADED by the independent method review: constant inputs + reused output
buffers could miss stale-epoch reads and silent no-op rounds. The detector
was hardened (3 rotating input banks + in-graph output poisoning + per-bank
references + a plain max-overlap variant; see docs/benchmark_method.md).
Hardened run (IEEE build, pdl=1): 50,000 distinguishable-data rounds/row +
1000 plain replays, zero mismatches. Fast-math build (the P0 deliverable),
same detector, pdl=1, 1000 replays: 50,000 in-graph bit-checked rounds per
row, instrumented mismatches = 0, plain final-check failures = 0, both rows
-> STABLE. P0 stability gate evidence complete for the shipped build.

## P0 Serving Gate (AC-6) — PASS (2026-07-10)

All runs: documented restart command; sanity = `benchmark_glm52_bs1.py
--runs 1` (40 greedy records; comparison = bench-recorded text fields:
completion_tokens, text_chars, text_prefix per task).

| Phase | Config | tok/s (overall decode) | Text records vs base |
|---|---|---:|---|
| base | resident long-warm server, pre-patch | 385.93 | (reference) |
| off | patch applied, flag unset | 376.53 | 40/40 identical |
| on (1st) | route ON, IEEE port build | 372.94 | 4/40 -> FAILED, diagnosed |
| bisect1 | route ON + stock module | 377.04 | 40/40 (route exonerated) |
| bisect2 | route ON, scope=attn, x2 runs | 372.07/372.43 | 4/40 vs base; 40/40 run-vs-run (deterministic) |
| on2 | stale IEEE build (patcher bug: early return skipped file refresh) | 374.11 | 4/40 — build fingerprint gate added after this |
| **on3** | **route ON, fast-math port (FTZ=14,334 verified)** | **376.27 (>= 376)** | **40/40 identical — GATE PASS** |
| restore/restore2 | flag unset | 376.70 / (in flight) | 40/40 — resident serving restored |

Default-OFF inertness: proven twice (off, restore). The integration patch
remains applied on the box with the flag unset between milestones (byte-inert
by measurement); full revert procedure is `solution/serving/sglang_patcher.py
revert`.

## P1 specialization (2026-07-10)

Latency decomposition that drove candidate selection (all on-box evidence):

| Measurement | T=6 | Meaning |
|---|---:|---|
| harness wall/round, 8-GPU concurrent 50-round graphs | 6.80-6.92 us | production convention; cross-round pipelining hides part of each call |
| serving in-graph per-call (profiler, SHAPES re-verify) | 8.3-8.5 us | the convention behind the <=7.0us target |
| solo eager single call, slots pre-fed (spin instant) | ~12 us eager / 16.3 us under NCU | full single-kernel span incl. eager launch mechanics |
| NCU (solo, T=6): SM throughput 0.91%, DRAM 0.62%, achieved occupancy 9.5%, no-eligible 91%, waves/SM 0.03 | — | the kernel is pure LATENCY: warps park on the Lamport wait; zero compute/bandwidth pressure |

Constraint that shapes the candidate space: the e2e promote gate requires
greedy output BYTE-IDENTICAL to the stock path, so candidates must preserve
every value-bearing operation, order, and the launch geometry (the norm's
cross-warp fp32 reduction tree depends on block geometry). Grid/block
re-configs and reduction restructuring from the original idea pool are
therefore value-breaking for THIS promote path and were not pursued past
analysis (recorded as out-of-scope routes, not attempted-and-failed).

### Candidate 1 — `oneshotArFusedNormConstKernel` (PROMOTED to e2e pass)

`solution/mnnvl_ar_fused/csrc/mnnvl_ar_fused_opt.cuh`: NumTokens/TokenDim as
template constants (index math folds, loops unroll; the norm division keeps
the RUNTIME tokenDim operand so the quotient bits stay identical) + the norm
operands (gamma, residual) prefetched BEFORE the Lamport spin so their load
latency hides behind the peer-arrival wait. Identical launch geometry via the
same adjustGridConfig. Dispatch: (T in {1,6}, H=6144, world=8, bf16,
rmsnorm, oneshot) -> constant instantiation; anything else falls back to the
generic verbatim dispatch inside the same entry (host-side scalar compares
only; no hot-path syncs).

Gates (all PASS):
- bf16 BIT-EXACT vs the flashinfer original, all 8 ranks, both outputs, both
  rows (promote-eligibility under byte-identity).
- Hardened stability, pdl=1, 1000 replays: 50,000 distinguishable-data
  rounds per row, zero mismatches; plain final-checks clean.
- Promotion A/B vs the ported P0 baseline (15 trials): T=6 1.0173, T=1
  1.0232, headline geomean 1.0202 — outside the measured noise envelope
  (|1-pseudo| up to 1.29%); no production row < 0.97x. 25-trial
  confirmation + fresh noise floor: see below.

NCU breakdown (solo pre-fed launch, T=6, `--replay-mode application`,
reports: `profile/p1_baseline/solo_{jit,opt}_t6.ncu-rep`):

| Metric | ported baseline (jit) | candidate (opt) |
|---|---:|---:|
| Duration under NCU | 16.29 us | 15.36 us (-5.7%) |
| Elapsed cycles | 17,777 | 16,746 |
| SM throughput | 0.91% | 0.95% |
| DRAM throughput | 0.62% | 0.66% |
| Registers/thread | 61 | 58 |
| Achieved occupancy | 9.49% | 9.44% |
| No-eligible | 91.28% | 91.13% |
| Grid x block | 24 x 192 (6 tokens x 4-block clusters) | same (geometry pinned for byte-identity) |

Named bound (why the remaining headroom is small under this contract): the
kernel is end-to-end latency-bound on the NVLS fan-out + Lamport arrival
wait (91% no-eligible at 0.9% SM utilization); value-preserving compute
optimizations only trim the post-arrival tail. The <=7.0us serving-convention
target status is measured in the promote pass (in-serving per-call profile).

### 25-trial confirmation & noise floor

| Row | jit median us | opt median us | speedup | fresh noise spread (opt vs opt, 10 trials) |
|---|---:|---:|---:|---:|
| T=6 | 6.831 | 6.726 | 1.0157 | 0.94% (std 0.117/0.124) |
| T=1 | 6.576 | 6.435 | 1.0219 | 0.14% (std 0.036/0.080) |

Headline geomean **1.0188** over the production rows — beyond the measured
noise envelope on both rows; both rows above 1.0 (row floor 0.97x holds with
margin). AC-7.1 hard promotion bar: **MET**. Pre-timing bit-exact ties held
on every benchmark invocation.

## Dispatch table

| Regime | Kernel | Selected by |
|---|---|---|
| T=6, H=6144, world=8, bf16, oneshot, rmsnorm | `oneshotArFusedNormConstKernel<8,bf16,6,6144>` | host scalar compare in `oneshotArFusedConstDispatch` |
| T=1, H=6144, world=8, bf16, oneshot, rmsnorm | `oneshotArFusedNormConstKernel<8,bf16,1,6144>` | same |
| any other shape/dtype/world/pattern | generic verbatim `oneshotAllreduceFusionKernel` dispatch | fallback inside the same entry |
| serving, payload > oneshot threshold (e.g. prefill) | stock flashinfer route (twoshot) | callsite route rejects, stock path runs |
| serving, `SGLANG_JIT_MNNVL_AR` unset/0 | stock flashinfer route | env gate default OFF |

Correctness is never lost: every uncovered combination reaches the verbatim
port or the stock flashinfer path.
