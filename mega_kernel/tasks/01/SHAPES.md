# 01_dense_fp8_gemm_bs1 — frozen shapes (per TP-rank, GLM-5.2 FP8 TP=8)

out[M,N] bf16 = A[M,K] bf16 @ dequant(W[N,K] fp8e4m3, S[⌈N/128⌉,⌈K/128⌉] f32)^T

M ∈ {1 (draft steps), 6 (verify / draft-extend)}. Weighted by per-iteration
call count (verify 78 layers dominates):

| shape (N×K) | role | calls/iter (M=6) | calls/iter (M=1) |
|---|---|---:|---:|
| 2624×6144 | fused q_a+kv_a | 79 | 4 |
| 2048×2048 | q_b | 79 | 4 |
| 6144×2048 | o_proj | 79 | 4 |
| 512×6144  | shared-expert gate_up | 76 | 4 |
| 6144×256  | shared-expert down | 76 | 4 |
| 3072×6144 | dense-MLP gate_up (layers 0-2) | 3 | 0 |
| 6144×1536 | dense-MLP down | 3 | 0 |
| 3584×512 / 4096×2048 / 128×6144 | indexer/misc (低频) | ~20 | ~2 |

## Baselines (BOTH must be beaten on the weighted mix)
1. DeepGEMM fp8 path incl. its per-call activation quant
   (`sglang_per_token_group_quant_fp8` + `deep_gemm.fp8_gemm_nt`): in-graph
   ≈ 8.7 + 2.3 µs per call at M=6.
2. cuBLAS bf16 (current production, `SGLANG_BS1_BF16_DENSE=1`): cold-L2
   measured M=6: 2624×6144 8.83 µs, 6144×2048 6.09, 512×6144 5.96,
   2048×2048 5.93, 6144×256 2.35, 3072×6144 9.45, 6144×1536 4.91.

## Targets / direction
- M=1: port the existing B200 winner (KDA-Pilot `deep_gemm_fp8_fp8_bf16_nt`
  M=1 GEMV, kernel 1.356×; B-scale register shuffle preload + split-K×2 for
  N≤3072) to sm_103a. Expected ≤3 µs/call.
- M=6: CUTLASS SM100 blockwise tensor-core (two prior CUDA-core attempts and
  one hand mma.m16n8k16 attempt are documented dead ends — see
  `glm52_blog_bench/k2/`). Floor: DRAM read at ≥5 TB/s + ~1.5 µs fixed
  → qkv_a ~4 µs, o_proj ~3.5 µs, weighted mean ~3.8 µs incl. no act-quant.
- Deterministic split-K only (bitwise-stable across replays).

## Serving hook (already in place)
`fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback` dispatch at M≤8
(env `SGLANG_BS1_TRITON_FP8_GEMV` slot — rename on promote) with plain-float
scale stash attached at load (`requant_weight_ue8m0` hook). Promoting this
task removes the need for `SGLANG_BS1_BF16_DENSE` → runtime returns to 100%
FP8 weights.

## Bench protocol
Cold-L2: 48 weight copies, all calls captured in one CUDA graph, report
us/call from graph replay (see `glm52_blog_bench/k2/k2_mma_test.py` harness).
Correctness: fp32 oracle rel < 2e-2; scale-ramp + ragged-N adversarial rows.
