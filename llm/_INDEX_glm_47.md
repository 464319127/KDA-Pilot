# glm_47 — e2e kernel task selection

- Model: `nvidia/GLM-4.7-NVFP4` (tp=8)
- Cookbook cmd: `python -m sglang.launch_server --model nvidia/GLM-4.7-NVFP4 --tp 8 --quantization modelopt_fp4 --reasoning-parser glm45 --trust-remote-code`
- Kept: max GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `glm_47__fp8_bmm` | quant_gemm | fp8_bmm | 19.5% | random_high | role |
| `glm_47__linear_gemm` | quant_gemm | linear_gemm | 16.5% | sharegpt_low | role |
| `glm_47__quant_fp8` | quant_gemm | quant_fp8 | 14.2% | random_low | role |
| `glm_47__sglang_unified_attention_with_output` | attention | attention | 4.9% | sharegpt_low | yes |
| `glm_47__rmsnorm` | norm | rmsnorm | 4.4% | sharegpt_mid | role |

## Dropped < 3.0%


## Excluded (comm / trtllm fused-MoE)

- ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelA (comm, comm): up to 24.8%
- void flashinfer::trtllm_allreduce_fusion::allreduce_fus (comm, comm): up to 20.3%
- void moe::dev::finalize::finalizeKernelVecLoad<moe::dev (moe, fused_moe_trtllm): up to 4.0%
- void moe::dev::routing::routingCustom::routingIndicesBl (moe, fused_moe_trtllm): up to 3.1%
- void moe::dev::finalize::finalizeKernel<moe::dev::final (moe, fused_moe_trtllm): up to 2.3%
