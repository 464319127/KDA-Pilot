# KDA Prompt: glm52_bs1__skinny_gemm_bf16_tc

Target GPU: NVIDIA B300 (sm_103). Beat cuBLAS/nvjet on the **fixed set** of
skinny bf16 GEMMs in GLM-5.2 bs=1 decode — **~14% of the round** (~2.4 ms).
Tensor cores mandatory: a careful CUDA-core GEMV already lost to nvjet at
every one of these shapes (prior art in `docs/prior_art/skinny_gemv.cu`).

## Problem (FIXED shape set, out = x @ W^T)

```
x [M, K] bf16, W [N, K] bf16, out [M, N] bf16, fp32 accumulate
M ∈ {1, 8} are the two weights that matter (chain steps / verify)
per-layer shapes (x78 layers):
  qkv_a     K=6144  N=2112      (q_a 1536 + kv_a 512+64, fused)
  q_b       K=1536  N=960
  o_proj    K=640   N=6144      (16 heads x 40 v-dim per rank... K fixed 640)
  sh_gateup K=6144  N=512       (shared expert gate+up fused)
  sh_down   K=256   N=6144
per-round shapes (x1):
  eh_proj   K=12288 N=6144      (MTP)
  lm_head   K=6144  N=18944     (vocab shard per rank; also 6 chain steps @M=1)
```

## Baseline

cuBLAS via `torch.nn.functional.linear` (dispatches to nvjet_sm103 kernels).
Measure it first (CUDA-graph timed) and record per-shape µs in
`docs/benchmark_method.md` — treat the recorded table as the contract.
Indicative total: ~30 µs/layer at M=8.

**Success: ≥ 1.25x geomean speedup over cuBLAS across the shape set at M=8,
≥ 1.15x at M=1, max rel err ≤ 2e-3 vs fp32 oracle.** A single kernel
template specialized per shape (constexpr K/N) is fine — shapes never change.

## Approach notes

- At M ≤ 8 these are memory-bound on W (e.g. lm_head: 232 MB → 29 µs
  roofline; qkv_a: 26 MB → 3.2 µs). nvjet already runs close-ish; wins come
  from (a) fusing the M-loop into one tensor-core tile (M=8 fits one
  tcgen05/wgmma instruction M-slice; nvjet pads to larger M tiles), (b)
  skipping the split-K reduce kernel nvjet launches for large K (fuse the
  reduction), (c) PDL chaining consecutive projections (qkv_a → q_b) inside
  one launch window, (d) `evict_first` streaming of W.
- Weight layout may be repacked offline (W is static): any tile-friendly
  layout is allowed as long as the repack is a one-time transform recorded in
  the adapter (production repacks at load).
- tcgen05 MMA with bf16 → fp32 TMEM accumulate, or wgmma sm_90 path if
  simpler — both compile for sm_103.

Tier B (accept-gated for main-model deploy; lm_head/eh_proj/MTP-internal
shapes are draft/head-side and safer). Follow
`../../llm/docs/llm_kernel_optimization_rules.md` +
`../../llm/docs/llm_correctness_contract.md`.
