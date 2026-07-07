# glm_47_flash — standalone kernel task selection

- Model: `zai-org/GLM-4.7-Flash` (tp=1)
- Serving capture cmd (provenance only): `sglang serve --model-path zai-org/GLM-4.7-Flash --tp 1 --attention-backend triton --reasoning-parser glm45 --tool-call-parser glm47`
- Task mode: standalone single-GPU kernel optimization; no live serve, run_capture, or multi-GPU e2e gate during RLCR.
- Kept: max serving-profile GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `glm_47_flash__sglang_unified_attention_with_output` | other | attention | 75.3% | sharegpt_mid | yes |
| `glm_47_flash__sglang_inplace_fused_experts` | moe | fused_moe_triton | 30.4% | random_mid | yes |
| `glm_47_flash__linear_gemm` | quant_gemm | linear_gemm | 15.6% | random_low | role |

## Dropped < 3.0%

- fused_add_rmsnorm: 2.8%

## Excluded (comm / trtllm fused-MoE)

