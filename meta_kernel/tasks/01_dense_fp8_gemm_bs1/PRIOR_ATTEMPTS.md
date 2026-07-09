# Documented dead ends (do not repeat)

1. Triton GEMV ×3 architectures (2026-07-08, B300 in-graph):
   - v1 single-program-per-strip: K-loop latency-bound, 26.6 µs @ 2624×6144
   - v2/v3 split-K 2-pass, dot-orientation flip: 14.2-16.6 µs — fixed cost
     ~4 µs/kernel + ~1.1 µs per 128-K iteration; loses to DeepGEMM+quant (11)
2. Hand mma.m16n8k16 (smem-staged dequant, cp.async double-buffer,
   `k2_mma_kernel.cu` here): WRONG (frag layout bug, rel~1.3) AND slow
   (25.6 µs vs cold cuBLAS bf16 8.83 µs @ 2624×6144). Not worth debugging —
   the structure (per-step smem roundtrip + 2×__syncthreads) can't reach BW.
3. B200 campaign CUDA-core M>1: 0.011-0.44× (two independent runs) → no-go.

Consensus direction: CUTLASS SM100/SM103 blockwise-scaled tensor-op GEMM with
small-M specialization (M padded 16), or extend DeepGEMM's Hopper swapAB
small-M path to Blackwell 1D1D templates.

Benchmark trap log: replay-same-weight microbench measures L2 (5-6 TB/s),
not DRAM; always rotate ≥48 weight copies (cold-L2 harness in k2_mma_test.py).
