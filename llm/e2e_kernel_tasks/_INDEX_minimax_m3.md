# minimax_m3 — e2e kernel task selection

- Model: `MiniMaxAI/MiniMax-M3-MXFP8` (tp=8)
- Cookbook cmd: `python -m sglang.launch_server --model-path MiniMaxAI/MiniMax-M3-MXFP8 --tp 8 --trust-remote-code --quantization mxfp8 --attention-backend fa4 --page-size 128 --mem-fraction-static 0.65`
- Kept: max GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `minimax_m3__linear_gemm` | quant_gemm | linear_gemm | 24.1% | random_high | role |
| `minimax_m3__rmsnorm` | norm | rmsnorm | 20.0% | random_low | role |
| `minimax_m3__mxfp8_block_scaled_matmul_kernel` | quant_gemm | mxfp8_block_scaled_matmul_kernel | 10.2% | sharegpt_low | role |
| `minimax_m3__fused_add_rmsnorm` | gemm | fused_add_rmsnorm | 3.7% | sharegpt_low | role |

## Dropped < 3.0%

- flashinfer_plan_kernel_int_int_i: 2.2%
- gqa_share_sparse_decode_kernel: 2.1%

## Excluded (comm / trtllm fused-MoE)

- ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelA (comm, comm): up to 24.4%
- void (anonymous namespace)::all_reduce_one_shot_push_ke (comm, comm): up to 24.2%
- void (anonymous namespace)::all_reduce_two_shot_kernel< (comm, comm): up to 9.1%
