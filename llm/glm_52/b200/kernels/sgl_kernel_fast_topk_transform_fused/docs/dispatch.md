# Candidate Dispatch & Design — fast_topk_transform_fused (B200)

> Status: **Bucket 1 IMPLEMENTED + correctness-verified (Round 9)**; design from task8 (Round 5,
> KernelWiki + Codex `analyze` ranking). `solution/candidate_topk_transform.cu` now contains a native
> CUDA decode copy/fill kernel for Bucket 1 with metadata-only dispatch + baseline fallback. On the
> frozen 251-row grid (B200 GPU 1, R9): **`matched_ratio = 1.0000 (251/251)`** with **221 calls taking
> the native kernel and 30 falling back** (the large-prefill/radix `min(N,M)>2048` rows) — definitive
> proof the native path executes and is exact. **Candidate-vs-baseline timing + per-bucket speedups +
> NCU active bound are still pending a strictly-idle GPU 1** (user chose to wait; only the benchmark is
> timing-gated — implementation/correctness are not). Speedups + `docs/results.md` filled after that.

## Implemented (Round 9) — Bucket 1 native decode copy/fill
- `solution/candidate_topk_transform.cu`: `fast_topk_transform_candidate` dispatches to
  `decode_copy_fill_kernel` when `topk==2048 && row_starts==None && S==B && min(N,M)<=2048` (+ int32
  dtype / contiguous-dst / `src.stride(1)==1` / CUDA guards); every other case calls
  `fast_topk_transform_interface` (the recovered baseline). The kernel writes all 2048 outputs/row,
  `dst[b,col] = (col<lengths[b]) ? src_page_table[b,col] : -1` (seq==b on this bucket), one CTA per
  (row, 256-col tile) and one thread/column (coalesced int32 stores; multiple CTAs/row for small-B
  occupancy). No `score` read, no host `lengths` read, no host sync, no hot-path allocation; launches on
  `at::cuda::getCurrentCUDAStream()`. An env-gated (`TOPK_CANDIDATE_DEBUG`) one-time diagnostic reports
  bucket-vs-fallback per call (used to confirm the 221/30 split; no-op by default).
- Correctness: naive bucket rows pass exact candidate==baseline==oracle; fallback rows keep the
  baseline's valid-top-k behavior. `bench/correctness.py` → 251/251 (R9, see `docs/run_log.md`).
- **Hardened (Round 10)** for final-publication robustness (correctness-preserving — re-verified
  251/251 with the SAME 221 native / 30 fallback split): cheap rank guards run FIRST so the size reads
  cannot throw on an out-of-contract shape (`score.dim==2`, `lengths.dim==1 && size==B`,
  `src_page_table.dim==2`, `cu_seqlens_q.dim==1 && size==B+1`) → any unexpected shape falls back to the
  baseline's own `TORCH_CHECK`s; and a non-synchronizing `C10_CUDA_KERNEL_LAUNCH_CHECK()` after the
  launch surfaces a launch-config error immediately (matches the baseline). Every memory access the
  kernel makes is now guarded.
- NOT yet done (timing-gated): candidate benchmark vs the frozen baseline, measured per-bucket speedup,
  NCU active-bound confirmation. These wait for a strictly-idle GPU 1.

## Regimes (from the captured contract)
- **Naive (`length = min(N,M) <= topk`)** — the baseline writes `dst[i] = (i<length)?src_page_table[i]:-1`,
  no score read, fully deterministic; **dominant: 3674/4246 captured calls**. Memory-bound (store of one
  `(B,2048)` int32 row per token = 8 KB/row).
- **Tiny-B decode** — `B in {2,6,8,14,18,20}`, the bulk of calls; many tiny launches → launch/occupancy bound.
- **Large-B prefill radix (`length > topk`)** — `B in {2207..3080}`, `N` up to 5151; ~572 calls incl. 4
  with `row_starts` tensors. The candidate needs only a **valid** top-k here (selected score multiset ==
  true top-k); it is NOT required to bit-match the non-deterministic baseline order.

## Ranked candidate directions (benefit vs risk; KernelWiki + Codex)
1. **Decode exact-naive tiled copy/fill** (FIRST EDIT). For `topk==2048 && row_starts==None && S==B &&
   min(N,M)<=2048`: one CTA per `(row, dst_tile)` (e.g. 8 tiles × 256) instead of a single 1024-thread row
   block; coalesced/vectorized (`int4`, 128-bit) `dst` stores; guarded page-table loads for `i<length`;
   vector `-1` fill for `[length,2048)`. Dominant regime, no score read, tiny correctness surface, exact-naive.
   Low risk. Attacks store throughput + low-CTA-count/tail underutilization.
2. **Scheme-X bucket tuning** (per TRT-LLM PR-13477 dispatcher): metadata buckets for tiny `B<=20` vs larger
   B, choosing CTAs-per-row + tile size separately. Attacks waves-per-SM/tail. Low-med risk, exact-naive.
3. **Extend exact-naive to prefill/ragged** (`S!=B` and/or `row_starts!=None`, `min(N,M)<=topk`): correct
   `cu_seqlens_q` token→seq mapping + row-start window. Medium risk.
4. **Native valid-radix (GVR)** for `min(N,M)>topk`: approximate bin to find the boundary, then verify/refine
   with full float so the selected multiset is the exact true top-k. High per-call benefit, low volume, high
   risk. KernelWiki basis: `pr-TensorRT-LLM-13477` (Guess-Verify-Refine).
5. **PDL / CUDA-Graph launch orchestration** for repeated `(B,N)` buckets / radix guess→verify→refine.
   Medium, integration-dependent.
6. **Memory micro-tuning** (128-bit vector stores/loads where aligned, `L1::no_allocate` streaming reads,
   `__launch_bounds__`/register budgeting). Apply after layout is right. KernelWiki: `pattern-memory-bound`.

## Active-bound hypotheses (confirm with NCU, AC-8/AC-10)
- Naive large-B: store/DRAM bound — expect low compute util, store-heavy L2/DRAM; win from coalesced/vectorized
  full-row writes + avoiding excess threads.
- Tiny-decode: launch/tail/under-occupancy bound — expect waves-per-SM << 1, low SM-active, DRAM not saturated;
  multi-CTA row tiling helps device time but cannot remove fixed launch overhead.

## Dispatch plan (metadata-only; no host read of `lengths`, no sync, no per-call cudaMalloc)
Inputs available without sync: `B, N, M, S, topk, row_starts pointer (null?), score.stride, dtype/device/alignment`.
- **Bucket 1 (implement first):** `topk==2048 && row_starts==None && S==B && min(N,M)<=2048` → native exact-naive decode.
- Future bucket 2: `... && S!=B && min(N,M)<=2048` → native exact-naive prefill.
- Future bucket 3: `... && row_starts!=None && min(N,M)<=2048` → native exact-naive ragged.
- Future bucket 4: `min(N,M)>2048 && score.stride(1)==1` (+ supported row/ragged mode) → native valid-radix/GVR.
- **Fallback rule:** any uncovered shape, unsupported `topk`, uncertain mapping, bad alignment, or
  not-yet-implemented path → the recovered SGLang baseline (`fast_topk_transform_interface`).

## Pitfalls (correctness-critical)
- Naive: write ALL 2048 outputs/row; `i>=length` is exactly `-1`; do NOT read `score`; non-contiguous score
  is irrelevant on this path. Use `min(N,M)` (N>2048 with M<=2048 is still naive). Do NOT host-check `lengths[b]`.
- `seq=b` only when `row_starts==None && S==B`; prefill maps token→seq via `cu_seqlens_q`; ragged `row_starts`
  offsets the score window (`score[idx+row_start]`) — the page-table index is still the relative position.
- Output entries are page-table VALUES (`src_page_table[seq][pos]`), not relative indices.
- Radix: output order may differ from the baseline, but the selected score multiset must equal the true top-k;
  8-bit bins alone are invalid — the boundary needs full-float refinement.
- Vectorized loads/stores need alignment/pitch checks; scalar tail or fallback otherwise.

## Correctness gate (already enforced, R3/R4)
`bench/correctness.py` must stay `matched_ratio==1.0` after every candidate edit: naive = exact
candidate==baseline==oracle; radix = valid-top-k (order/tie tolerant) with row_starts + real page-table inversion.
