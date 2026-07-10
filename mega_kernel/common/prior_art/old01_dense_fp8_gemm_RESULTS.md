# 01 — sm_103 (B300) port status, 2026-07-09

## M=1 GEMV port (from B200 winner `decode_gemv.cu`)

Compiles clean with `-gencode=arch=compute_103a,code=sm_103a` via torch
cpp_extension; no source changes needed. Devbox `glm52-bs1-opt` GPU7 (server
co-resident, idle between requests).

Correctness: 8/8 shapes OK vs fp32 oracle (rel ≤ 2.7e-3, exact-pow2 ue8m0
scales).

Cold-L2 in-graph (48-case rotation, graph replay, µs/call):

| N×K | candidate | note |
|---|---:|---|
| 6144×256 | (in corr. run only) | |
| 4096×2048 | 6.25 | |
| 6144×2048 | 6.74 | 12.6 MB → 1.87 TB/s |
| 512×6144 | 8.10 | splitK=2 path |
| 2624×6144 | 9.09 | 16.1 MB → 1.77 TB/s |
| 3072×6144 | 9.91 | |

Reference: DeepGEMM same shapes in-server ≈ 8.7 µs (M-independent) → port is
~1.0-1.4× at M=1 on sm103, consistent with the B200 kernel-honest 1.356×
result but NOT near DRAM BW — the warp-per-column design is memory-LATENCY
bound (single outstanding 512B load/warp), same as documented on B200.

deep_gemm in-graph baseline in this harness errors (heuristics assertion on
hand-packed ue8m0 layout) — baseline number taken from serving profile
instead; fix the packing to their TMA-aligned util if a same-harness A/B is
needed for promotion.

## Integration reality check

At bs=1 serving, M=1 dense GEMMs occur only in the draft path (~20
calls/iter): even a 2× kernel is worth <0.1 ms/iter e2e. The e2e money in
this task is the **M=6 verify path (~390 calls/iter)** = CUTLASS SM100/103
blockwise tensor-op small-M kernel (see PRIOR_ATTEMPTS.md for the three
documented dead ends). M=1 port serves as the pipeline validation leg.

## Next
1. CUTLASS blockwise-scaled GEMM prototype at M=16-pad (the real deliverable).
2. Optional: bf16-activation variant of the M=1 GEMV (drops the act-quant
   kernel; changes interface to A bf16 + f32 scales — matches the serving
   hook's stashed-scale convention).
