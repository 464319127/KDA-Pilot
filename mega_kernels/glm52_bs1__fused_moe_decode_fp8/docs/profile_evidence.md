# Profile evidence — glm52_bs1__fused_moe_decode_fp8

**e2e target: ~23% of the 17.6 ms decode round** (~4.1 ms) on the mini-sglang
GLM-5.2 bs=1 deployment at 292.6 tok/s (B300 x8, TP=8, k=7 MTP spec decode,
commit `a26fd6f`).

- Per layer @ verify M=8: triton gemm1+gemm2 ≈ 47 µs (tuned in-server) +
  glue ≈ 8 µs (post-`tiny_align`), × 75 MoE layers.
- Also runs at M=1 in each of the 6 chained MTP draft steps and at M=1..8 in
  the MTP extend (1 MoE layer each).
- Every µs saved at M=8 ≈ 75 µs/round ≈ +1.2 tok/s.

## Fixed call signature (from live capture, mini-sglang `glm_moe_dsa.py`)

```
x            [M, 6144]        bf16, contiguous          M ∈ 1..8
gate_up      [256, 512, 6144] float8_e4m3fn, contiguous ([gate;up] rows)
gu_scale_inv [256, 4, 48]     fp32
down         [256, 6144, 256] float8_e4m3fn
dn_scale_inv [256, 48, 2]     fp32
topk_ids     [M, 8] int32 (distinct per row), topk_w [M, 8] fp32
out          [M, 6144] bf16
```

Real block scales span ~1e-4..1e-1 (sample log-uniformly in the oracle).
Expert-id distribution: verify batches touch ~50-64 distinct experts of 256.

## Measured baselines (CUDA-graph timed, idle B300, this exact shape set)

| impl | M=1 | M=4 | M=8 |
|---|---:|---:|---:|
| triton fused_experts_impl (standalone, default cfg) | 41.7 | 73.8 | 101 |
| prior art moe_decode.cu v5 (g1+g2 split @M=8: 43+23) | 19.3 | 41.6 | 66 |
| DRAM roofline | ~5 | ~20 | ~37 |

## Reproduce e2e (needs the B300 node + weights)

```bash
MINISGL_SPEC_STEPS=7 MINISGL_GRAPH_BS=1,2,3,4,5,6,7,8 MINISGL_MTP_HIDDEN=post \
python -m minisgl --model-path /data/bbuf/glm52_real --tp-size 8 \
  --attention-backend mla --memory-ratio 0.85 --page-size 64 \
  --cuda-graph-max-bs 8 --cache-type naive --port 30000
# bench: /data/bbuf/bench_bs1_30000.py ; accept: MINISGL_SPEC_LOG=1 histogram
```

Kernel-level success is standalone; e2e re-run validates deployment tier B.
