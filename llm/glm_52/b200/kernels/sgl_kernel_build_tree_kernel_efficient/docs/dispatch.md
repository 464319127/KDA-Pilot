# Dispatch Table

## Regime analysis
The captured GLM-5.2 B200 interface is a single fixed scalar point across all 187
variants: `topk=1, depth=1, draft_token_num=2, tree_mask_mode=FULL_MASK`,
contiguous int64 inputs + bool `tree_mask`, `parent_list` shape `[bs, 0]` (empty).
Only `bs` (1..10) and `T = 2*sum(verified_seq_len) + 4*bs` vary; the op's work is
`bs`-determined and `T`-independent. So "per-shape dispatch" collapses to ONE
specialized fast path + a baseline fallback — no multi-bucket family is warranted.

## Dispatch logic (host-side, no device sync on the hot path)
`solution/build_tree_candidate.cu :: build_tree_candidate` (TVM-FFI `TensorView`
ABI) selects, checking ALL metadata that defines the fast-path domain via
`DLDataType` + a contiguity helper (`bte::`):

| Condition (all must hold) | Path |
|---|---|
| scalars `topk==1 && depth==1 && draft_token_num==2 && tree_mask_mode==FULL_MASK` AND dtypes (parent_list/selected_index/verified_seq_len/positions/retrive_* int64, tree_mask bool) AND shapes (`parent_list [bs,0]`, `selected_index [bs,1]`, `verified_seq_len [bs]`, outputs numel `bs*2`) AND all contiguous AND all CUDA | **candidate** fast kernel |
| anything else (other scalar mode, draft≠2, non-empty/non-int64 parent_list, wrong selected_index shape/dtype, non-contiguous, non-CUDA) | **baseline** fallback |

`parent_list.size(1)==0` is the sync-free signal of the degenerate depth-1 regime
where `selected_index` must be all-zero, so the fast path skips parent traversal.
No output values are read on the host → no device synchronization. Fallback routing
is verified by 5 correctness cases (QLEN_ONLY, non-contig vsl, parent_list int32,
selected_index `[bs,2]`, draft=4) — each matches the baseline bit-for-bit.

## Per-regime baseline-vs-candidate (controlled same-process probe, all bs 1–10)
Controlled probe (`bench/floor_probe.py` Section A; 31 trials, tight p10/p90; fresh
retrieve buffers per call → no reuse artifact). Removes the cross-subprocess clock
noise of the official harness; this is the conclusive per-regime verdict.

| bs | floor µs | baseline µs | candidate µs | speedup | verdict | chosen path |
|---|---|---|---|---|---|---|
| 1 | 3.36 | 9.42 | 9.54 | 0.987 | tie | candidate(==baseline) |
| 2 | 3.35 | 9.47 | 9.52 | 0.995 | tie | candidate(==baseline) |
| 3 | 3.36 | 9.45 | 9.56 | 0.988 | tie | candidate(==baseline) |
| 4 | 3.35 | 9.52 | 9.63 | 0.989 | tie | candidate(==baseline) |
| 5 | 3.36 | 9.46 | 9.61 | 0.985 | tie | candidate(==baseline) |
| 6 | 3.36 | 9.41 | 9.62 | 0.979 | tie | candidate(==baseline) |
| 7 | 3.35 | 9.46 | 9.62 | 0.984 | tie | candidate(==baseline) |
| 8 | 3.35 | 9.45 | 9.61 | 0.984 | tie | candidate(==baseline) |
| 9 | 3.31 | 9.42 | 9.59 | 0.982 | tie | candidate(==baseline) |
| 10 | 3.31 | 9.37 | 9.53 | 0.983 | tie | candidate(==baseline) |
| fallback (off-domain) | — | == baseline by construction | — | ~1.00 | baseline |

Controlled geomean 0.9854; official-harness production geomean 0.9932. **Every bs is
a statistical tie** — no kernel-level win survives the ~3.35µs launch floor + ~6µs
8-tensor `TensorView` marshalling that dominate the <0.1µs kernel body.

## Conclusion
**No-go under the strict op ABI**: a single specialized fast path + fallback is the
correct shape, the candidate is correct on every regime, but it ties the baseline
(host-overhead/launch-bound). The only material lever — fusing the wrapper prefill
(~63% of the realistic path, measured) — is out of promotion scope per DEC-1. See
`docs/results.md`.
