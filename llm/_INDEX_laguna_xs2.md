# laguna_xs2 — e2e kernel task selection

- Model: `poolside/Laguna-XS.2-NVFP4` (tp=1)
- Cookbook cmd: `python -m sglang.launch_server --model-path poolside/Laguna-XS.2-NVFP4 --tp 1 --trust-remote-code`
- Kept: max GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `laguna_xs2__linear_gemm` | quant_gemm | linear_gemm | 42.4% | sharegpt_high | role |
| `laguna_xs2__sglang_unified_attention_with_output` | attention | attention | 11.3% | random_high | yes |
| `laguna_xs2__void_cvt_fp16_to_fp4_nv_bfloat16` | quant_gemm | void_cvt_fp16_to_fp4_nv_bfloat16 | 9.2% | random_mid | role |
| `laguna_xs2__compute_expert_blockscale_offset` | moe | compute_expert_blockscale_offset | 7.1% | random_low | role |
| `laguna_xs2__void_at_native_elementwise_kerne` | memory_bound | void_at_native_elementwise_kerne | 4.6% | random_mid | role |

## Dropped < 3.0%

- void_anonymous_namespace_fused_q: 3.0%
- void_cublas_lt_split_kreduce_ker: 2.9%
- fused_add_rmsnorm: 2.9%
- activation: 2.8%
- void_apply_shuffle_mul_sum_kerne: 2.7%
- rmsnorm: 2.2%
- void_at_native_unrolled_elementw: 2.2%

## Excluded (comm / trtllm fused-MoE)

