# Dispatch Table

## Regime analysis
The captured GLM-5.2 B200 interface is a single fixed scalar point across all 187
variants: `topk=1, depth=1, draft_token_num=2, tree_mask_mode=FULL_MASK`,
contiguous int64 inputs + bool `tree_mask`, `parent_list [bs, 0]` (empty). Only
`bs` (1..10) and `T = 2*sum(verified_seq_len) + 4*bs` vary; the op's work is
`bs`-determined and `T`-independent. So a single specialized fast path + baseline
fallback is the right shape — no multi-bucket family is warranted.

## Dispatch logic (host-side, no device sync on the hot path)
`solution/build_tree_candidate.cu` shares one predicate
(`candidate_fast_path_eligible`) between the dispatcher and the route diagnostic.
It selects, via `DLDataType` + a zero-numel-safe contiguity helper (`bte::`):

| Condition (all must hold) | Path |
|---|---|
| scalars `topk==1 && depth==1 && draft_token_num==2 && tree_mask_mode==FULL_MASK` AND dtypes (parent_list/selected_index/verified_seq_len/positions/retrive_* int64, tree_mask bool) AND shapes (`parent_list [bs,0]`, `selected_index [bs,1]`, `verified_seq_len [bs]`, outputs numel `bs*2`) AND all contiguous AND the dereferenced tensors (verified_seq_len + the 5 outputs) on CUDA | **candidate** fast kernel |
| anything else | **baseline** fallback |

`parent_list.size(1)==0` is the sync-free signal of the degenerate depth-1 regime
(selected_index all-zero), so the fast path skips parent traversal. No output
values are read on the host → no device synchronization.

## Route coverage — PROVEN (not just output equality)
A TVM-FFI diagnostic `build_tree_candidate_route` (1=fast / 0=fallback, no launch)
runs the same predicate; `bench/correctness.py` asserts:
- **route==1 (native fast path) for every captured production row, all bs 1..10**
  (incl. bs=10 `parent_list [10,0]` — the Round-1 contiguity bug returned 0 here and
  silently fell back; now fixed: zero-element tensors are contiguous).
- **route==0 (baseline fallback) for all 5 off-domain cases** (non-FULL_MASK,
  non-contiguous vsl, parent_list int32, selected_index `[bs,2]`, draft=4).

## Candidate fast path
One block, one thread per request, draft_token_num fixed to 2; writes only the
deltas vs the True/-1 pre-state (`tree_mask[S_b+L_b+1]=false`; positions;
retrive_index; `retrive_next_token[2b]=1`). Matches the recovered baseline's net
effect bit-for-bit.

## Per-regime baseline-vs-candidate (controlled probe, all bs 1–10; candidate genuinely runs)
`bench/floor_probe.py` Section A (31 trials, tight p10/p90, fresh retrieve buffers):

| bs | floor µs | baseline µs | candidate µs | speedup | verdict | chosen path |
|---|---|---|---|---|---|---|
| 1 | 6.71 | 12.60 | 12.64 | 0.997 | tie | candidate |
| 2 | 4.57 | 10.77 | 10.79 | 0.999 | tie | candidate |
| 3 | 4.36 | 10.59 | 10.71 | 0.989 | tie | candidate |
| 4 | 4.46 | 10.60 | 10.84 | 0.977 | tie | candidate |
| 5 | 5.20 | 10.68 | 10.95 | 0.976 | tie | candidate |
| 6 | 4.79 | 10.93 | 11.04 | 0.990 | tie | candidate |
| 7 | 4.86 | 10.90 | 11.03 | 0.988 | tie | candidate |
| 8 | 4.84 | 10.89 | 11.03 | 0.988 | tie | candidate |
| 9 | 4.75 | 10.93 | 11.11 | 0.984 | tie | candidate |
| 10 | 4.84 | 10.90 | 11.02 | 0.990 | tie | candidate |
| fallback (off-domain) | — | == baseline by construction | — | ~1.00 | baseline |

Controlled geomean 0.9877; official-harness production geomean 0.9839 (0 clean wins,
22 real regressions, 161 ties). Every bs is a tie-to-mild-regression — no
kernel-level win survives the ~5µs launch floor + multi-arg marshalling.

## Conclusion
**No-go under the strict op ABI**: the candidate is route-proven to run on the full
captured regime, is correct everywhere, but ties-to-mildly-regresses (host/launch-
bound; kernel body is a sub-µs sliver of ~10–12µs). The baseline is retained as the
promoted path. The only material lever — wrapper-prefill fusion (~64%, measured) —
is out of promotion scope per DEC-1. See `docs/results.md`.
