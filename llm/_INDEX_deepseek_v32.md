# deepseek_v32 — e2e kernel task selection

- Model: `nvidia/DeepSeek-V3.2-NVFP4` (tp=4)
- Cookbook cmd: `python -m sglang.launch_server --model nvidia/DeepSeek-V3.2-NVFP4 --tp 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3`
- Kept: max GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `deepseek_v32__sglang_fp4_gemm` | quant_gemm | linear_gemm | 32.5% | sharegpt_low | yes |
| `deepseek_v32__fp8_bmm` | quant_gemm | fp8_bmm | 21.2% | random_high | role |
| `deepseek_v32__sglang_unified_attention_with_output` | attention | attention | 13.3% | sharegpt_mid | yes |
| `deepseek_v32__void_anonymous_namespace_fast_ha` | other | void_anonymous_namespace_fast_ha | 5.1% | sharegpt_mid | role |
| `deepseek_v32__quant_fp8` | quant_gemm | quant_fp8 | 3.2% | random_low | role |

## Dropped < 3.0%

- fused_add_rmsnorm: 2.4%
- rope: 2.2%

## Excluded (comm / trtllm fused-MoE)

- void flashinfer::trtllm_allreduce_fusion::allreduce_fus (comm, comm): up to 18.7%
- ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelA (comm, comm): up to 18.0%
- void moe::dev::finalize::finalizeKernelVecLoad<moe::dev (moe, fused_moe_trtllm): up to 2.6%
