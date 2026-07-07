# llama33_70b — standalone kernel task selection

- Model: `meta-llama/Llama-3.3-70B-Instruct` (tp=1)
- Serving capture cmd (provenance only): `sglang serve --model-path meta-llama/Llama-3.3-70B-Instruct --tp 1 --tool-call-parser llama3`
- Task mode: standalone single-GPU kernel optimization; no live serve, run_capture, or multi-GPU e2e gate during RLCR.
- Kept: max serving-profile GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `llama33_70b__linear_gemm` | quant_gemm | linear_gemm | 86.6% | random_mid | role |
| `llama33_70b__sglang_unified_attention_with_output` | attention | attention | 3.6% | sharegpt_mid | yes |

## Dropped < 3.0%

- rmsnorm: 2.7%
- fused_add_rmsnorm: 2.3%

## Excluded (comm / trtllm fused-MoE)

