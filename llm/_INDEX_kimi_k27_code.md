# kimi_k27_code — e2e kernel task selection

- Model: `moonshotai/Kimi-K2.7-Code` (tp=8)
- Cookbook cmd: `sglang serve --model-path moonshotai/Kimi-K2.7-Code --tp 8 --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --trust-remote-code`
- Kept: max GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `kimi_k27_code__linear_gemm` | quant_gemm | linear_gemm | 38.4% | sharegpt_low | role |
| `kimi_k27_code__fp8_bmm` | quant_gemm | fp8_bmm | 28.4% | sharegpt_mid | role |

## Dropped < 3.0%

- attention: 2.7%

## Excluded (comm / trtllm fused-MoE)

- void flashinfer::trtllm_allreduce_fusion::allreduce_fus (comm, comm): up to 36.1%
- void moe::dev::finalize::finalizeKernelVecLoad<moe::dev (moe, fused_moe_trtllm): up to 2.1%
