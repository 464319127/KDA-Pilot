# ring_26_1t B200 KDA kernel launchers

One-click KDA task launchers for the ring_26_1t (`inclusionAI/Ring-2.6-1T`) B200 kernel
optimization tasks under `llm/ring_26_1t/b200/kernels/`. Same flow and rules as
`llm/glm_52/b200/scripts/` (CUDA-only candidate, warp-spec profiling, per-shape
dispatch, round-robin GPU pin). The launcher draft mandates the shared contract
docs under `llm/docs/` before any implementation.

## Usage

```bash
# Launch one kernel task (creates a worktree + starts Claude Code):
llm/ring_26_1t/b200/scripts/launch_kernels/k01_b200_activation_run_activation_inplace.sh

# Prepare the worktree + draft without launching Claude (dry run):
KDA_NO_CLAUDE=1 llm/ring_26_1t/b200/scripts/launch_kernels/<kNN_...>.sh

# Launch any kernel folder directly:
llm/ring_26_1t/b200/scripts/launch_kda_kernel_task.sh llm/ring_26_1t/b200/kernels/<dir>
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
| 01 | `k01_b200_activation_run_activation_inplace` | 0 | `jit_kernel_activation_run_activation_inplace` |
| 02 | `k02_b200_rope_apply_rope_inplace` | 1 | `jit_kernel_rope_apply_rope_inplace` |
| 03 | `k03_b200_bmm_fp8` | 2 | `sgl_kernel_bmm_fp8` |
| 04 | `k04_b200_fp8_scaled_mm` | 3 | `sgl_kernel_fp8_scaled_mm` |
| 05 | `k05_b200_fused_add_rmsnorm` | 4 | `sgl_kernel_fused_add_rmsnorm` |
| 06 | `k06_b200_moe_align_block_size` | 5 | `sgl_kernel_moe_align_block_size` |
| 07 | `k07_b200_moe_sum_reduce` | 6 | `sgl_kernel_moe_sum_reduce` |
| 08 | `k08_b200_rmsnorm` | 7 | `sgl_kernel_rmsnorm` |
| 09 | `k09_b200_sgl_per_token_quant_fp8` | 0 | `sgl_kernel_sgl_per_token_quant_fp8` |
| 10 | `k10_b200_quant_method_compressed_tensors_linear_method_apply` | 1 | `sglang_quant_method_compressed_tensors_linear_method_apply` |
| 11 | `k11_b200_moe_moe_runner_triton_utils_fused_moe_inplace_fused_experts` | 2 | `srt_layers_moe_moe_runner_triton_utils_fused_moe_inplace_fused_experts` |
| 12 | `k12_b200_moe_topk_fused_topk_deepseek` | 3 | `srt_layers_moe_topk_fused_topk_deepseek` |
| 13 | `k13_b200_deepseek_common_attention_forward_methods_forward_mla_bmm_fp8_op` | 4 | `srt_models_deepseek_common_attention_forward_methods_forward_mla_bmm_fp8_op` |

## Excluded: communication kernels (no launcher)

Communication kernels are not optimized here, so no launcher is generated for:

- `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce`
- `srt_distributed_parallel_state_inplace_all_reduce`
- `srt_distributed_parallel_state_outplace_all_reduce`
- `srt_distributed_parallel_state_reg_all_gather_into_tensor`
