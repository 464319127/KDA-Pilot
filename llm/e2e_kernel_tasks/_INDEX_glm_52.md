# glm_52 — e2e kernel task selection

- Model: `zai-org/GLM-5.2-FP8` (tp=8)
- Cookbook cmd: `python -m sglang.launch_server --model-path zai-org/GLM-5.2-FP8 --tp 8 --trust-remote-code --mem-fraction-static 0.8`
- Kept: max GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `glm_52__sglang_deep_gemm_fp8_fp8_bf16_nt` | quant_gemm | linear_gemm | 28.6% | sharegpt_low | yes |
| `glm_52__fp8_bmm` | quant_gemm | fp8_bmm | 25.0% | sharegpt_mid | role |
| `glm_52__sglang_unified_attention_with_output` | attention | attention | 14.2% | sharegpt_mid | yes |
| `glm_52__per_token_group_quant` | quant_gemm | per_token_group_quant | 10.9% | random_low | role |

## Dropped < 3.0%

- fused_add_rmsnorm: 2.0%

## Excluded (comm / trtllm fused-MoE)

- void flashinfer::trtllm_allreduce_fusion::allreduce_fus (comm, comm): up to 34.4%
- void (anonymous namespace)::all_reduce_one_shot_push_ke (comm, comm): up to 2.6%
- void moe::dev::finalize::finalizeKernelVecLoad<moe::dev (moe, fused_moe_trtllm): up to 2.2%
- void moe::dev::finalize::finalizeKernel<moe::dev::final (moe, fused_moe_trtllm): up to 2.2%
- void moe::dev::routing::routingCustom::routingIndicesBl (moe, fused_moe_trtllm): up to 2.1%
