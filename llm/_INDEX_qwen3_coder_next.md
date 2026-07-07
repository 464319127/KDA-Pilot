# qwen3_coder_next — standalone kernel task selection

- Model: `Qwen/Qwen3-Coder-Next` (tp=2)
- Serving capture cmd (provenance only): `sglang serve --model-path Qwen/Qwen3-Coder-Next --tp 2 --tool-call-parser qwen3_coder`
- Task mode: standalone single-GPU kernel optimization; no live serve, run_capture, or multi-GPU e2e gate during RLCR.
- Kept: max serving-profile GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `qwen3_coder_next__linear_gemm` | quant_gemm | linear_gemm | 40.5% | random_low | role |
| `qwen3_coder_next__fp8_bmm` | other | fp8_bmm | 15.6% | random_mid | role |

## Dropped < 3.0%


## Excluded (comm / trtllm fused-MoE)

- void flashinfer::trtllm_allreduce_fusion::allreduce_fus (comm, comm): up to 34.4%
- _fwd_kernel (other, comm): up to 7.4%
- bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x64x128u2_s5_et1 (other, comm): up to 4.6%
- ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelA (comm, comm): up to 4.4%
- void moe::dev::finalize::finalizeKernel<moe::dev::final (moe, fused_moe_trtllm): up to 3.3%
- void moe::dev::finalize::finalizeKernelVecLoad<moe::dev (moe, fused_moe_trtllm): up to 3.1%
- void moe::dev::routing::routingCustom::routingIndicesBl (moe, fused_moe_trtllm): up to 2.6%
