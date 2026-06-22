# minimax_m3 B200 KDA kernel launchers

One-click KDA task launchers for the minimax_m3 (`MiniMaxAI/MiniMax-M3-MXFP8`) B200 kernel
optimization tasks under `llm/minimax_m3/b200/kernels/`. Same flow and rules as
`llm/glm_52/b200/scripts/` (CUDA-only candidate, warp-spec profiling, per-shape
dispatch, round-robin GPU pin). The launcher draft mandates the shared contract
docs under `llm/docs/` before any implementation.

## Usage

```bash
# Launch one kernel task (creates a worktree + starts Claude Code):
llm/minimax_m3/b200/scripts/launch_kernels/k01_b200_flash_attention_v4_flash_attn_varlen_func.sh

# Prepare the worktree + draft without launching Claude (dry run):
KDA_NO_CLAUDE=1 llm/minimax_m3/b200/scripts/launch_kernels/<kNN_...>.sh

# Launch any kernel folder directly:
llm/minimax_m3/b200/scripts/launch_kda_kernel_task.sh llm/minimax_m3/b200/kernels/<dir>
```

## GPU assignment (round-robin 0-7)

Each wrapper pins its task to a B200 GPU, cycling `(N-1) % 8` so the
kernels spread across the 8 cards. Override per run with `KDA_GPU_ID=<id>`.

```
k01->0  k02->1  k03->2  k04->3  k05->4  k06->5  k07->6  k08->7  k09->0  k10->1  k11->2
```

## Kernels with launchers (11 compute kernels)

| # | launcher | GPU | kernel task dir |
|---|---|---|---|
| 01 | `k01_b200_flash_attention_v4_flash_attn_varlen_func` | 0 | `jit_kernel_flash_attention_v4_flash_attn_varlen_func` |
| 02 | `k02_b200_flash_attention_v4_flash_attn_with_kvcache` | 1 | `jit_kernel_flash_attention_v4_flash_attn_with_kvcache` |
| 03 | `k03_b200_kvcache_store_cache` | 2 | `jit_kernel_kvcache_store_cache` |
| 04 | `k04_b200_minimax_qknorm_rope_fused_gemma_qknorm_rope` | 3 | `jit_kernel_minimax_qknorm_rope_fused_gemma_qknorm_rope` |
| 05 | `k05_b200_moe_fused_gate_moe_fused_gate` | 4 | `jit_kernel_moe_fused_gate_moe_fused_gate` |
| 06 | `k06_b200_gemma_fused_add_rmsnorm` | 5 | `sgl_kernel_gemma_fused_add_rmsnorm` |
| 07 | `k07_b200_gemma_rmsnorm` | 6 | `sgl_kernel_gemma_rmsnorm` |
| 08 | `k08_b200_quant_method_fp8_linear_method_apply` | 7 | `sglang_quant_method_fp8_linear_method_apply` |
| 09 | `k09_b200_attention_base_attn_backend_attention_backend_forward` | 0 | `srt_layers_attention_base_attn_backend_attention_backend_forward` |
| 10 | `k10_b200_quantization_fp8_utils_triton_mxfp8_block_scaled_matmul` | 1 | `srt_layers_quantization_fp8_utils_triton_mxfp8_block_scaled_matmul` |
| 11 | `k11_b200_quantization_fp8_utils_triton_mxfp8_blockscaled_linear` | 2 | `srt_layers_quantization_fp8_utils_triton_mxfp8_blockscaled_linear` |

## Excluded: communication kernels (no launcher)

Communication kernels are not optimized here, so no launcher is generated for:

- `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce`
- `srt_distributed_parallel_state_inplace_all_reduce`
- `srt_distributed_parallel_state_outplace_all_reduce`
- `srt_distributed_parallel_state_reg_all_gather_into_tensor`
