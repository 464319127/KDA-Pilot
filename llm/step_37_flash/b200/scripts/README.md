# step_37_flash B200 KDA kernel launchers

One-click KDA task launchers for the step_37_flash (`stepfun-ai/Step-3.7-Flash-FP8`) B200 kernel
optimization tasks under `llm/step_37_flash/b200/kernels/`. Same flow and rules as
`llm/glm_52/b200/scripts/` (CUDA-only candidate, warp-spec profiling, per-shape
dispatch, round-robin GPU pin). The launcher draft mandates the shared contract
docs under `llm/docs/` before any implementation.

## Usage

```bash
# Launch one kernel task (creates a worktree + starts Claude Code):
llm/step_37_flash/b200/scripts/launch_kernels/k01_b200_activation_run_activation_filtered_inplace.sh

# Prepare the worktree + draft without launching Claude (dry run):
KDA_NO_CLAUDE=1 llm/step_37_flash/b200/scripts/launch_kernels/<kNN_...>.sh

# Launch any kernel folder directly:
llm/step_37_flash/b200/scripts/launch_kda_kernel_task.sh llm/step_37_flash/b200/kernels/<dir>
```

## GPU assignment (round-robin 0-7)

Each wrapper pins its task to a B200 GPU, cycling `(N-1) % 8` so the
kernels spread across the 8 cards. Override per run with `KDA_GPU_ID=<id>`.

```
k01->0  k02->1  k03->2  k04->3  k05->4  k06->5  k07->6  k08->7  k09->0  k10->1  k11->2  k12->3  k13->4
```

## Kernels with launchers (13 compute kernels)

| # | launcher | GPU | kernel task dir |
|---|---|---|---|
| 01 | `k01_b200_activation_run_activation_filtered_inplace` | 0 | `jit_kernel_activation_run_activation_filtered_inplace` |
| 02 | `k02_b200_activation_run_activation_inplace` | 1 | `jit_kernel_activation_run_activation_inplace` |
| 03 | `k03_b200_kvcache_store_cache` | 2 | `jit_kernel_kvcache_store_cache` |
| 04 | `k04_b200_rope_apply_rope_inplace` | 3 | `jit_kernel_rope_apply_rope_inplace` |
| 05 | `k05_b200_gemma_fused_add_rmsnorm` | 4 | `sgl_kernel_gemma_fused_add_rmsnorm` |
| 06 | `k06_b200_gemma_rmsnorm` | 5 | `sgl_kernel_gemma_rmsnorm` |
| 07 | `k07_b200_moe_align_block_size` | 6 | `sgl_kernel_moe_align_block_size` |
| 08 | `k08_b200_moe_sum_reduce` | 7 | `sgl_kernel_moe_sum_reduce` |
| 09 | `k09_b200_sgl_per_token_group_quant_8bit` | 0 | `sgl_kernel_sgl_per_token_group_quant_8bit` |
| 10 | `k10_b200_topk_sigmoid` | 1 | `sgl_kernel_topk_sigmoid` |
| 11 | `k11_b200_quant_method_unquantized_linear_method_apply` | 2 | `sglang_quant_method_unquantized_linear_method_apply` |
| 12 | `k12_b200_attention_base_attn_backend_attention_backend_forward` | 3 | `srt_layers_attention_base_attn_backend_attention_backend_forward` |
| 13 | `k13_b200_moe_moe_runner_triton_utils_fused_moe_inplace_fused_experts` | 4 | `srt_layers_moe_moe_runner_triton_utils_fused_moe_inplace_fused_experts` |

## Excluded: communication kernels (no launcher)

Communication kernels are not optimized here, so no launcher is generated for:

- `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce`
- `srt_distributed_parallel_state_inplace_all_reduce`
- `srt_distributed_parallel_state_outplace_all_reduce`
- `srt_distributed_parallel_state_reg_all_gather_into_tensor`
