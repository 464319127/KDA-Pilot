# Results: `sgl_kernel.build_tree_kernel_efficient` (GLM-5.2, B200)

Run on remote host `ion-b200`, NVIDIA B200 GPU id 6 (idle before and after).
Upstream baseline commit `7e6587c94a1d0305815a14067c5d3cc02a9b0f36`. tvm_ffi 0.1.9,
torch+cu13, compiled for `sm_100`, symmetric `-std=c++17 -O3` (no fast-math).
Candidate source sha256: `32abc09611f5a475370f07dac71b0e3b84468f8c13e60382c08db0afec3b647a`
(`solution/build_tree_candidate.cu`).

## Verdict: evidence-backed NO-GO (under the strict op ABI, DEC-1)
The native-CUDA candidate is **correct** but provides **no statistically stable,
regression-free speedup** over the recovered baseline, so it fails the DEC-2
promotion bar ("any statistically stable, regression-free geomean win > 1.0").
The recovered baseline is retained as the recommended path; the candidate stays in
`solution/` as the reasoned native-CUDA attempt behind this no-go.

Named active bound: **ABI + launch-floor bound**. Each call pays the ~3.35µs
empty-kernel launch floor plus ~6µs of TVM-FFI / multi-arg / metadata dispatch (an
inferred delta vs a 1-tensor noop); the actual kernel body is **<0.1µs of the
~9.5µs** per-call time. Reducing grid blocks (the candidate's lever) therefore buys nothing
measurable. The op writes only O(bs) integers + one `tree_mask` flip per request
(a few KB) — far below any compute/bandwidth limit; it is launch/marshalling-bound,
not compute-bound. Warp-specialization does not apply (no producer/consumer GEMM).

## Correctness — PASS
`bench/correctness.py`: **691 cases, 0 failures**. Baseline AND candidate match an
independent Python oracle bit-for-bit (exact int64/bool/tree) across the 183
production rows × {uniform, skewed, monotonic, random} `verified_seq_len`
distributions, the full captured `(bs,T)` range sweep, poison + required pre-state,
separate in-place copies, and 5 fallback cases (QLEN_ONLY mode, non-contiguous
`verified_seq_len`, `parent_list` int32, `selected_index [bs,2]`, `draft_token_num=4`)
— each routes to the baseline and matches it exactly. The official benchmark's
inline exact-match gate passed all 187 workloads.

## Benchmark — official harness, ALL 183 production rows
- Equal-weight production geomean speedup = **0.9932** (arith mean 0.9933,
  min 0.9474, max 1.0391, n=183). 128/183 rows below 1.0.
- Classifying each production row by non-overlapping p10/p90: **1** real win,
  **5** real "regressions", **177** ties. Per-bs median speedup: bs1 0.989,
  bs2 0.999, bs3 0.994, bs4 0.994, bs5 0.998, bs6 0.998, bs7 1.009, bs8 1.003,
  bs9 0.988, bs10 0.986. Absolute: baseline 14.7–19.6µs, candidate 14.7–19.1µs.
- The 5 sub-noise "regressions" are cross-subprocess GPU-clock noise (the
  controlled same-process probe below shows every bs a tie). Net: a statistical
  tie, very slightly negative — **not** a regression-free win.

## Controlled same-process probe (secondary diagnostic, removes subprocess clock noise)
`bench/floor_probe.py` Section A — both sides timed back-to-back, 31 trials, tight
p10/p90, ALL bs 1–10 (each measured invocation uses fresh -1 retrieve buffers from
a ring, so the baseline takes its if-branch every call — no reuse artifact):

| bs | floor µs | baseline µs | candidate µs | speedup | verdict |
|---|---|---|---|---|---|
| 1 | 3.36 | 9.42 | 9.54 | 0.987 | tie |
| 2 | 3.35 | 9.47 | 9.52 | 0.995 | tie |
| 3 | 3.36 | 9.45 | 9.56 | 0.988 | tie |
| 4 | 3.35 | 9.52 | 9.63 | 0.989 | tie |
| 5 | 3.36 | 9.46 | 9.61 | 0.985 | tie |
| 6 | 3.36 | 9.41 | 9.62 | 0.979 | tie |
| 7 | 3.35 | 9.46 | 9.62 | 0.984 | tie |
| 8 | 3.35 | 9.45 | 9.61 | 0.984 | tie |
| 9 | 3.31 | 9.42 | 9.59 | 0.982 | tie |
| 10 | 3.31 | 9.37 | 9.53 | 0.983 | tie |

geomean 0.9854; **every bs (incl. 7 and 9) is a statistical tie** — no kernel-level
win survives the ~6µs marshalling + ~3.35µs launch floor. This is the conclusive,
clock-noise-free verdict.

## Wrapper-inclusive diagnostic (MEASURED — DEC-1 secondary, NOT promoted)
`bench/floor_probe.py` Section B — the real per-call cost the captured Python
callsite pays for the op's outputs (`tree_mask.fill_(True)` + `retrieve_buf =
torch.full((3,bs,2), -1)` + `positions = torch.empty(...)` + op), vs the op alone:

| bs | T | op_only µs | wrapper-incl µs | prefill_add µs | op fraction |
|---|---|---|---|---|---|
| 4 | 2246 | 11.21 | 29.72 | 18.51 | 0.377 |
| 10 | 5382 | 10.74 | 29.53 | 18.79 | 0.364 |
| 10 | 11626 | 10.68 | 29.36 | 18.68 | 0.364 |

The wrapper prefill is **~18.6µs ≈ 63%** of the realistic path; the op is only
~36–38%. This (now measured, not inferred) confirms the only material latency lever
is fusing the wrapper prefill into the kernel — which changes the measured ABI
boundary and is **out of promotion scope per DEC-1**, recorded as an opt-in
recommendation for a future wider integration patch.

## Per-shape dispatch
See `docs/dispatch.md`. Single specialized fast path (fixed regime) + baseline
fallback; the candidate routes all off-domain shapes/dtypes/scalars/contiguity to
the baseline (verified by the 5 fallback correctness cases).

## Independent cross-check (Codex, task12)
An independent Codex reassessment of the full-coverage results **upheld the NO-GO**:
the candidate is not a stable, regression-free geomean win (official 0.9932 /
controlled 0.9854; all bs ties), so claiming even a marginal win would require
cherry-picking. Refinements adopted above: (1) the bound is an **ABI + launch-floor
bound** — ~3.35µs empty-kernel floor + ~6µs TVM-FFI/multi-arg/metadata dispatch
(an inferred delta vs a 1-tensor noop) over a `<0.1µs` kernel body — not "host
overhead" alone; (2) the official 5 "real regressions" are weak labels (7 samples
in a launch-bound op) and are treated as promotion blockers, not proof of
deterministic slowdown — the 31-trial same-process probe finds no reproducible
per-bs regression. Codex confirmed no material optimization remains under the
strict op ABI (DEC-1); the only lever is wrapper/prefill fusion (~63%, out of
scope), with analogous SGLang precedent for reducing Python-side small-kernel
overhead (PR-6369, PR-5086). Recommendation: keep the recovered baseline as the
promoted path; retain the candidate as evidence of the exhausted strict-ABI attempt.

## Provenance
- Host / GPU id 6 / model / before+after idle + versions / exact commands: `docs/run_log.md`.
- Baseline commit + copied files: `docs/baseline_source.md`.
- ABI / flags / timing / ring / workloads: `docs/benchmark_method.md`.
- Raw per-run records: `bench/results.jsonl`; controlled-probe output: `bench/floor_probe_out.txt` (results.jsonl gitignored).
