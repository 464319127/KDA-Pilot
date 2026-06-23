# Results: `sgl_kernel.build_tree_kernel_efficient` (GLM-5.2, B200)

Run on remote host `ion-b200`, NVIDIA B200 GPU id 6 (idle before and after).
Upstream baseline commit `7e6587c94a1d0305815a14067c5d3cc02a9b0f36`. tvm_ffi build,
`sm_100`, symmetric `-std=c++17 -O3` (no fast-math). Candidate source sha256:
`70e6a38853f01d160d0d1214700db46fd113d7bd7bcf58bc2c4d707ba43750f5`.

## Verdict: definitive NO-GO (strict op ABI, DEC-1) — candidate genuinely exercised
With the Round-1 dispatch bug fixed, the native-CUDA candidate now **provably runs**
on every captured production row (route==1 for all bs 1..10; see Route coverage),
and it still produces **no statistically stable, regression-free speedup** — it ties
to mildly regresses (geomean 0.984 official / 0.988 controlled; **0 clean wins**).
It fails the DEC-2 promotion bar. The recovered baseline is retained as the
recommended path; the candidate stays in `solution/` as the reasoned native-CUDA
attempt behind this no-go.

Named active bound: **host / launch / marshalling bound**. Empty-kernel launch
floor ≈ 4.4–6.7µs; baseline and candidate both ≈ 10.6–12.6µs; the kernel body is a
sub-µs sliver of that. There is essentially no kernel headroom, so no kernel-body
design can produce a stable win. (The candidate's single-block, one-thread-per-
request layout is in fact marginally *worse* than the baseline's bs-block parallel
launch for bs>1 — hence the mild regression rather than an exact tie; a bs-block
redesign could at best reach a tie, not a win.) The only material lever is the
wrapper prefill (~64% of the realistic path, measured), out of promotion scope per
DEC-1.

## Route coverage (proves the candidate path is actually taken)
A TVM-FFI route diagnostic (`build_tree_candidate_route`, 1=fast / 0=fallback, no
launch) runs the SAME predicate as `build_tree_candidate`. `bench/correctness.py`
asserts it for every case:
- **Native fast path (route==1) for every captured production row, all bs 1..10**
  — including bs=10 `parent_list [10,0]` (the Round-1 contiguity bug returned 0 here
  and silently fell back; now fixed: zero-element tensors are contiguous).
- **Baseline fallback (route==0) for all 5 off-domain cases** (non-FULL_MASK mode,
  non-contiguous `verified_seq_len`, `parent_list` int32, `selected_index [bs,2]`,
  `draft_token_num=4`).
This closes the Round-1 gap where output-equality could pass even though the
candidate had silently fallen back.

## Correctness — PASS
`bench/correctness.py`: **691 cases, 0 failures** (183 production × distributions +
full `(bs,T)` sweep + 5 fallback cases) — baseline AND candidate exact-match an
independent oracle, plus the route assertions above.

## Benchmark — official harness, ALL 183 production rows (candidate genuinely running)
- Equal-weight production geomean = **0.9839** (arith 0.9842, min 0.787, max 1.133).
  150/183 rows below 1.0.
- By non-overlapping p10/p90: **0 real wins, 22 real regressions, 161 ties**.
- Per-bs median speedup (all < 1.0): bs1 0.987, bs2 0.978, bs3 0.981, bs4 0.984,
  bs5 0.994, bs6 0.979, bs7 0.981, bs8 0.971, bs9 0.983, bs10 0.995. Absolute:
  baseline 10.8–24.9µs, candidate 10.9–25.6µs.

## Controlled same-process probe (all bs 1..10; candidate truly runs)
`bench/floor_probe.py` Section A (31 trials, tight p10/p90, fresh retrieve buffers):

| bs | floor µs | baseline µs | candidate µs | speedup | verdict |
|---|---|---|---|---|---|
| 1 | 6.71 | 12.60 | 12.64 | 0.997 | tie |
| 2 | 4.57 | 10.77 | 10.79 | 0.999 | tie |
| 3 | 4.36 | 10.59 | 10.71 | 0.989 | tie |
| 4 | 4.46 | 10.60 | 10.84 | 0.977 | tie |
| 5 | 5.20 | 10.68 | 10.95 | 0.976 | tie |
| 6 | 4.79 | 10.93 | 11.04 | 0.990 | tie |
| 7 | 4.86 | 10.90 | 11.03 | 0.988 | tie |
| 8 | 4.84 | 10.89 | 11.03 | 0.988 | tie |
| 9 | 4.75 | 10.93 | 11.11 | 0.984 | tie |
| 10 | 4.84 | 10.90 | 11.02 | 0.990 | tie |

geomean 0.9877; **every bs a tie, zero clean wins** — consistent with the official
run. No kernel-level win survives the ~5µs launch floor + multi-arg marshalling.

## Wrapper-inclusive diagnostic (MEASURED — DEC-1 secondary, NOT promoted)
`bench/floor_probe.py` Section B: op_only ~12.1–12.3µs, wrapper-inclusive ~34.1–34.5µs,
prefill_add ~21.8–22.3µs (op fraction ~0.35) → the Python-side prefill
(`tree_mask.fill_(True)` + `retrieve = full(-1)`) is **~64%** of the realistic path.
Fusing it is the only material lever, and it changes the measured ABI boundary
(out of promotion scope per DEC-1; opt-in recommendation for a wider patch).

## Per-shape dispatch
See `docs/dispatch.md`. Single specialized fast path + baseline fallback; route
coverage proven (fast path all bs 1..10; fallback for off-domain).

## Independent cross-check (Codex, task-r2-verdict)
An independent Codex reassessment **upheld the final NO-GO as legitimate**: with the
candidate genuinely exercised (route-proven for all bs incl. `[10,0]`), the evidence
is sufficient — 183 production rows, correctness passes, geomean 0.9839, 0 real wins,
22 real regressions, per-bs median < 1.0 for every bs, controlled probe agrees. It
confirmed: (1) reporting the mild regression honestly is correct — no redesign to a
clean tie is needed because there is no promotion path under DEC-1 (a tie-polishing
rewrite's best case is "no worse," not faster); (2) the host/launch/wrapper ceiling
is right — with floor ~4.4–6.7µs and op ~10–12µs the kernel body is too small to
support a stable kernel-only win, and the real lever is wrapper/prefill fusion (~64%,
out of DEC-1 scope); (3) the methodology (route proof + assertions, 691-case
correctness, full 183-row benchmark, symmetric TVM-FFI, controlled probe) is sound
and changes nothing. Rational endpoint: retain the baseline, record the native fast
path as rejected, document the ceiling.

## Provenance
- Host / GPU id 6 / model / before+after idle + versions / commands: `docs/run_log.md`.
- Baseline commit + copied files: `docs/baseline_source.md`.
- ABI / flags / timing / ring / workloads / route diagnostic: `docs/benchmark_method.md`.
- Raw per-run records: `bench/results.jsonl`; controlled probe: `bench/floor_probe_out.txt` (results.jsonl gitignored).
