# intern_s2_preview B200 KDA kernel launchers

One-click KDA task launchers for the intern_s2_preview (`internlm/Intern-S2-Preview-FP8`) B200 kernel
optimization tasks under `llm/intern_s2_preview/b200/kernels/`. Same flow and rules as
`llm/glm_52/b200/scripts/` (CUDA-only candidate, warp-spec profiling, per-shape
dispatch, round-robin GPU pin). The launcher draft mandates the shared contract
docs under `llm/docs/` before any implementation.

## Usage

```bash
# Launch one kernel task (creates a worktree + starts Claude Code):
llm/intern_s2_preview/b200/scripts/launch_kernels/k01_b200_activation_run_activation_inplace.sh

# Prepare the worktree + draft without launching Claude (dry run):
KDA_NO_CLAUDE=1 llm/intern_s2_preview/b200/scripts/launch_kernels/<kNN_...>.sh

# Launch any kernel folder directly:
llm/intern_s2_preview/b200/scripts/launch_kda_kernel_task.sh llm/intern_s2_preview/b200/kernels/<dir>
```

## GPU assignment (round-robin 0-7)

Each wrapper pins its task to a B200 GPU, cycling `(N-1) % 8` so the
kernels spread across the 8 cards. Override per run with `KDA_GPU_ID=<id>`.

```
k01->0  k02->1  k03->2  k04->3  k05->4  k06->5  k07->6  k08->7  k09->0  k10->1
```

## Kernels with launchers (10 compute kernels)

| # | launcher | GPU | kernel task dir |
|---|---|---|---|
| 01 | `k01_b200_activation_run_activation_inplace` | 0 | `jit_kernel_activation_run_activation_inplace` |
| 02 | `k02_b200_kvcache_store_cache` | 1 | `jit_kernel_kvcache_store_cache` |
| 03 | `k03_b200_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2` | 2 | `jit_kernel_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2` |
| 04 | `k04_b200_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2_custom_op` | 3 | `jit_kernel_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2_custom_op` |
| 05 | `k05_b200_gemma_fused_add_rmsnorm` | 4 | `sgl_kernel_gemma_fused_add_rmsnorm` |
| 06 | `k06_b200_gemma_rmsnorm` | 5 | `sgl_kernel_gemma_rmsnorm` |
| 07 | `k07_b200_quant_method_fp8_linear_method_apply` | 6 | `sglang_quant_method_fp8_linear_method_apply` |
| 08 | `k08_b200_quant_method_unquantized_linear_method_apply` | 7 | `sglang_quant_method_unquantized_linear_method_apply` |
| 09 | `k09_b200_moe_flashinfer_trtllm_moe_trtllm_fp8_block_scale_moe_wrapper` | 0 | `srt_layers_moe_flashinfer_trtllm_moe_trtllm_fp8_block_scale_moe_wrapper` |
| 10 | `k10_b200_quantization_fp8_kernel_deep_gemm_fp8_fp8_bf16_nt` | 1 | `srt_layers_quantization_fp8_kernel_deep_gemm_fp8_fp8_bf16_nt` |

## Excluded: communication kernels (no launcher)

Communication kernels are not optimized here, so no launcher is generated for:

- `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce`
- `srt_distributed_parallel_state_inplace_all_reduce`
- `srt_distributed_parallel_state_outplace_all_reduce`
- `srt_distributed_parallel_state_reg_all_gather_into_tensor`
