# ernie45 — standalone kernel task selection

- Model: `baidu/ERNIE-4.5-21B-A3B-PT` (tp=1)
- Serving capture cmd (provenance only): `sglang serve --model-path baidu/ERNIE-4.5-21B-A3B-PT --tp 1`
- Task mode: standalone single-GPU kernel optimization; no live serve, run_capture, or multi-GPU e2e gate during RLCR.
- Kept: max serving-profile GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `ernie45__fused_moe_triton` | moe | fused_moe_triton | 65.0% | random_high | role |
| `ernie45__linear_gemm` | quant_gemm | linear_gemm | 29.1% | sharegpt_low | role |
| `ernie45__sglang_unified_attention_with_output` | attention | attention | 8.0% | random_high | yes |
| `ernie45__fused_add_rmsnorm` | gemm | fused_add_rmsnorm | 4.5% | sharegpt_low | role |

## Dropped < 3.0%

- rope: 2.4%
- rmsnorm: 2.2%
- moe_align_block_size: 2.2%

## Excluded (comm / trtllm fused-MoE)

