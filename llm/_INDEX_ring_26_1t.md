# ring_26_1t — standalone kernel task selection

- Model: `inclusionAI/Ring-2.6-1T` (tp=8)
- Serving capture cmd (provenance only): `sglang serve --model-path inclusionAI/Ring-2.6-1T --tp-size 8 --trust-remote-code --mem-fraction-static 0.8 --tool-call-parser glm --reasoning-parser deepseek-r1`
- Task mode: standalone single-GPU kernel optimization; no live serve, run_capture, or multi-GPU e2e gate during RLCR.
- Kept: max serving-profile GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `ring_26_1t__sglang_inplace_fused_experts` | moe | fused_moe_triton | 31.2% | random_mid | yes |
| `ring_26_1t__fused_add_rmsnorm` | gemm | fused_add_rmsnorm | 28.9% | sharegpt_low | role |
| `ring_26_1t__sgl_kernel_fp8_scaled_mm` | gemm | linear_gemm | 16.9% | random_high | yes |
| `ring_26_1t__sgl_kernel_sgl_per_token_quant_fp8` | quant_gemm | quant_fp8 | 4.2% | random_high | yes |

## Dropped < 3.0%

- void_moe_sum_reduce_warp_per_tok: 2.9%

## Excluded (comm / trtllm fused-MoE)

- void (anonymous namespace)::all_reduce_one_shot_push_ke (comm, comm): up to 32.1%
