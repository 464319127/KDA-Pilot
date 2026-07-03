# KDA Prompt: glm52_bs1__fused_moe_decode_fp8

Target GPU: NVIDIA B300 (sm_103). Build a fused block-fp8 MoE decode kernel for
GLM-5.2 bs=1 speculative decode — **~23% of the round** (~4.1 ms of 17.6 ms) on
the mini-sglang deployment (292.6 tok/s baseline). Fixed shapes, tier B
(accept-gated; MTP-layer deploy is unconditionally safe).

## Problem (all shapes FIXED)

One call replaces the whole routed-experts pipeline
(act-quant → moe_align → gemm1 → silu*mul → act-quant → gemm2 → weighted-sum):

```
inputs : x [M, 6144] bf16              M ∈ {1,2,3,4,5,6,7,8}, each M matters
         gate_up [256, 512, 6144] fp8e4m3  (w1w3 = [gate;up] row order)
         gu_scale [256, 4, 48] fp32        ([128,128] block dequant scales)
         down [256, 6144, 256] fp8e4m3
         dn_scale [256, 48, 2] fp32
         topk_ids [M, 8] int32, topk_w [M, 8] fp32  (weights pre-normalized)
output : out [M, 6144] bf16 = Σ_s w[t,s] · down_e ( silu(gate_e x) * (up_e x) )
```

## Baselines (measured, CUDA-graph timed, idle B300)

| impl | M=1 | M=4 | M=8 | notes |
|---|---:|---:|---:|---|
| sglang triton fused_experts_impl (w8a8 + glue) | 41.7 µs | 73.8 µs | 101 µs | standalone default config; in-server tuned ≈ 47 µs gemms + ~8 µs glue at M=8 |
| prior art `moe_decode.cu` (mini-sglang) | 19.3 µs | 41.6 µs | 66 µs | bf16-act × fp8-dequant, cp.async double-buffer; rel err 3e-3 vs fp32 oracle (triton: 4e-2) |
| DRAM roofline (≈50-64 distinct experts touched) | ~5 µs | ~20 µs | ~37 µs | 8 TB/s HBM3e |

**Success: M=8 ≤ 48 µs and M=1 ≤ 14 µs** at rel err ≤ 5e-3 vs the fp32-dequant
oracle (beats in-server tuned triton+glue end to end). Stretch: M=8 ≤ 42 µs.

## Where the headroom is (from prior-art autopsy, `docs/prior_art/`)

- gemm1 (gate_up, 200 MB reads @M=8) stalls at 4.7 TB/s with per-warp uint4
  prefetch; the cp.async warp-private variant was no better (chunks too small).
  Block-cooperative cp.async staging (like gemm2's, which hit its roofline
  share) or tcgen05/wgmma fp8 tensor-core tiles are the two obvious attacks —
  at M=8 a [16, 512] × [512, 6144] fp8 MMA per expert-group is tensor-core
  friendly if tokens are grouped by expert (grouping also removes the ~12%
  duplicate expert reads the per-pair prior art pays).
- Consider one persistent kernel over both gemms with PDL between phases; the
  intermediate is only [M, 8, 256] bf16 (32 KB).
- Weights are read-once: use `evict_first` / streaming hints, keep L2 for x
  and the intermediate.

## Deployment tiers

- MTP draft layer: ship freely (prior art already shipped there, accept
  5.0→5.2).
- Main model: requires the in-server accept A/B (see `_INDEX` "Shared
  benchmark seeds"): accept histogram bimodality must survive. Do NOT assume
  correctness ⇒ deployable; GSM8K stays 93-95% even when accept collapses.

Follow `../../llm/docs/llm_kernel_optimization_rules.md` (native CUDA
candidate) and `../../llm/docs/llm_correctness_contract.md`. Baseline source =
mini-sglang `python/minisgl/kernel/csrc/jit/moe_decode.cu` +
sglang `fused_experts_impl` (copy both per the rules; record SHAs in
`docs/baseline_source.md`).
