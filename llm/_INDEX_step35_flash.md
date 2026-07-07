# step35_flash — standalone kernel task selection

- Model: `stepfun-ai/Step-3.5-Flash` (tp=4)
- Serving capture cmd (provenance only): `sglang serve --model-path stepfun-ai/Step-3.5-Flash --tp 4 --trust-remote-code --reasoning-parser step3p5`
- Task mode: standalone single-GPU kernel optimization; no live serve, run_capture, or multi-GPU e2e gate during RLCR.
- Kept: max serving-profile GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `step35_flash__sglang_inplace_fused_experts` | moe | fused_moe_triton | 16.1% | random_mid | yes |
| `step35_flash__linear_gemm` | quant_gemm | linear_gemm | 6.9% | sharegpt_high | role |
| `step35_flash__void_moe_top_k_256_float_const_b` | moe | void_moe_top_k_256_float_const_b | 3.3% | sharegpt_high | role |
| `step35_flash__sgl_kernel_gemma_rmsnorm` | norm | rmsnorm | 3.1% | sharegpt_high | yes |

## Dropped < 3.0%


## Excluded (comm / trtllm fused-MoE)

- void sglang::cross_device_reduce_1stage<__nv_bfloat16,  (memory_bound, comm): up to 49.3%
