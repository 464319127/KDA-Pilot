# llama4_scout — e2e kernel task selection

- Model: `meta-llama/Llama-4-Scout-17B-16E-Instruct` (tp=8)
- Cookbook cmd: `python -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8 --trust-remote-code --mem-fraction-static 0.8 --context-length 65536`
- Kept: max GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `llama4_scout__fused_add_rmsnorm` | gemm | fused_add_rmsnorm | 34.8% | random_low | role |
| `llama4_scout__sglang_inplace_fused_experts` | moe | fused_moe_triton | 16.7% | random_mid | yes |
| `llama4_scout__linear_gemm` | quant_gemm | linear_gemm | 7.3% | sharegpt_low | role |

## Dropped < 3.0%

- attention: 2.4%

## Excluded (comm / trtllm fused-MoE)

- void (anonymous namespace)::all_reduce_one_shot_push_ke (comm, comm): up to 37.5%
- void (anonymous namespace)::all_reduce_two_shot_kernel< (comm, comm): up to 37.1%
- ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelA (comm, comm): up to 32.5%
