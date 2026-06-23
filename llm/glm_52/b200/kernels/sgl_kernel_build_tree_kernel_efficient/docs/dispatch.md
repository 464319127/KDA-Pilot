# Dispatch Table

## Regime analysis
The captured GLM-5.2 B200 interface is a single fixed scalar point across all 187
variants: `topk=1, depth=1, draft_token_num=2, tree_mask_mode=FULL_MASK`,
contiguous int64 inputs + bool `tree_mask`, with `parent_list` shape `[bs, 0]`
(empty). The only varying dimensions are `bs` (1..10) and the bool `tree_mask`
length `T = 2*sum(verified_seq_len) + 4*bs`. The op's work is determined by `bs`
and is independent of `T`. Therefore the expected "per-shape kernel dispatch"
collapses to ONE specialized fast path plus a baseline fallback — no multi-bucket
kernel family is warranted by the evidence (a single fixed-scalar point, only the
launch-bound `bs` axis varies).

## Dispatch logic (host-side, no device sync on the hot path)
`solution/build_tree_candidate.cu :: build_tree_candidate` selects:

| Condition (all must hold) | Chosen path |
|---|---|
| `topk==1 && depth==1 && draft_token_num==2 && tree_mask_mode==FULL_MASK(0)` AND `parent_list.dim()==2 && parent_list.size(1)==0` AND all tensors int64 (tree_mask bool) AND all contiguous | **candidate** fast kernel (`build_tree_candidate_kernel`) |
| anything else (other scalar mode, draft_token_num≠2, non-empty parent_list, wrong dtype, non-contiguous) | **baseline** fallback (`build_tree_baseline`, the recovered upstream op) |

`parent_list.size(1)==0` is the sync-free signal that we are in the degenerate
depth-1 regime where `selected_index` must be all-zero (any nonzero value would
dereference the zero-column `parent_list` in the baseline, i.e. an invalid input),
so the fast path can skip the parent traversal entirely. No output values are read
on the host, so the dispatch performs no device synchronization.

## Candidate fast path
One block (bs ≤ 256), one thread per request `b`, draft_token_num fixed to 2.
Each thread writes only what differs from the True/-1 pre-state:
`tree_mask[S_b + L_b + 1] = false`; `positions[2b]=L_b, [2b+1]=L_b+1`;
`retrive_index[2b]=2b, [2b+1]=2b+1`; `retrive_next_token[2b]=1`. It leaves
`retrive_next_token[2b+1]` and both `retrive_next_sibling` entries at their `-1`
pre-state, and all other `tree_mask` entries at their `True` pre-state — matching
the recovered baseline's net effect bit-for-bit. Versus the baseline (grid=bs
blocks × 2 threads), the candidate uses a single block to reduce grid scheduling;
it cannot reduce the host launch count below one launch.

## Per-regime baseline-vs-candidate results
Official harness, NVIDIA B200 GPU 6, CUDA-event median µs (see `docs/results.md`
for the noise caveat: the bs6/7/9 subprocesses ran in a low-clock state, so their
ratios understate the candidate; the controlled same-process probe shows
non-overlapping p10/p90 candidate-faster for bs 2–10).
| bs (regime) | representative T | baseline (µs) | candidate (µs) | speedup | chosen path |
|---|---|---|---|---|---|
| 1 | 300 | 4.122 | 2.887 | 1.428 | candidate |
| 2 | 1234 | 4.120 | 2.879 | 1.431 | candidate |
| 3 | 3814 | 4.120 | 2.955 | 1.394 | candidate |
| 4 | 2246 | 4.114 | 3.036 | 1.355 | candidate |
| 5 | 5528 | 4.117 | 4.046 | 1.017 | candidate |
| 6 | 6216 | 5.272 | 5.264 | 1.001 | candidate |
| 7 | 6024 | 5.264 | 5.387 | 0.977* | candidate |
| 8 | 9138 | 4.119 | 4.064 | 1.014 | candidate |
| 9 | 9842 | 5.304 | 5.402 | 0.982* | candidate |
| 10 | 5382 | 4.128 | 4.093 | 1.008 | candidate |
| fallback (QLEN_ONLY / non-contiguous) | — | == baseline by construction | — | ~1.00 | baseline |

`*` clock artifact (both sides ran ~5.3µs in a low-clock subprocess), not a real
regression — see `docs/results.md`. Production geomean = **1.144×** (official) /
**1.021×** (controlled same-process probe). Active bound is the CUDA launch floor
(~2.0–3.2µs, ~50–80% of runtime), so a single specialized fast path + baseline
fallback is the right shape; no multi-bucket kernel family is warranted.
