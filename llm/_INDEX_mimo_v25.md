# mimo_v25 — standalone kernel task selection

- Model: `XiaomiMiMo/MiMo-V2.5` (tp=4)
- Serving capture cmd (provenance only): `python -m sglang.launch_server --model-path XiaomiMiMo/MiMo-V2.5 --tp 4 --trust-remote-code --attention-backend fa4 --mm-attention-backend fa4 --moe-runner-backend flashinfer_trtllm --mem-fraction-static 0.65 --chunked-prefill-size 16384`
- Task mode: standalone single-GPU kernel optimization; no live serve, run_capture, or multi-GPU e2e gate during RLCR.
- Kept: max serving-profile GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `mimo_v25__fused_add_rmsnorm` | gemm | fused_add_rmsnorm | 34.6% | sharegpt_low | role |
| `mimo_v25__fp8_bmm` | quant_gemm | fp8_bmm | 27.9% | sharegpt_mid | role |
| `mimo_v25__linear_gemm` | gemm | linear_gemm | 9.2% | sharegpt_mid | role |
| `mimo_v25__rmsnorm` | gemm | rmsnorm | 3.9% | random_low | role |

## Dropped < 3.0%

- attention: 2.9%

## Excluded (comm / trtllm fused-MoE)

- void (anonymous namespace)::all_reduce_one_shot_push_ke (comm, comm): up to 36.8%
- void moe::dev::activation::activationDeepSeekKernel<moe (moe, fused_moe_trtllm): up to 3.2%
