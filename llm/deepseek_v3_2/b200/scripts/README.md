# deepseek_v3_2 B200 KDA kernel launchers

One-click KDA task launchers for the deepseek_v3_2 (`nvidia/DeepSeek-V3.2-NVFP4`) B200 kernel
optimization tasks under `llm/deepseek_v3_2/b200/kernels/`. Same flow and rules as
`llm/glm_52/b200/scripts/` (CUDA-only candidate, warp-spec profiling, per-shape
dispatch, round-robin GPU pin). The launcher draft mandates the shared contract
docs under `llm/docs/` before any implementation.

## Usage

```bash
# Launch one kernel task (creates a worktree + starts Claude Code):
llm/deepseek_v3_2/b200/scripts/launch_kernels/k01_b200_activation_run_activation_inplace.sh

# Prepare the worktree + draft without launching Claude (dry run):
KDA_NO_CLAUDE=1 llm/deepseek_v3_2/b200/scripts/launch_kernels/<kNN_...>.sh

# Launch any kernel folder directly:
llm/deepseek_v3_2/b200/scripts/launch_kda_kernel_task.sh llm/deepseek_v3_2/b200/kernels/<dir>
```

## GPU assignment (round-robin 0-7)

Each wrapper pins its task to a B200 GPU, cycling `(N-1) % 8` so the
kernels spread across the 8 cards. Override per run with `KDA_GPU_ID=<id>`.

```
k01->0  k02->1  k03->2  k04->3  k05->4  k06->5  k07->6  k08->7  k09->0  k10->1  k11->2  k12->3  k13->4  k14->5  k15->6  k16->7
```

## Kernels with launchers (16 compute kernels)

| # | launcher | GPU | kernel task dir |
|---|---|---|---|
| 01 | `k01_b200_activation_run_activation_inplace` | 0 | `jit_kernel_activation_run_activation_inplace` |
| 02 | `k02_b200_fused_store_index_cache_fused_store_index_k_cache` | 1 | `jit_kernel_fused_store_index_cache_fused_store_index_k_cache` |
| 03 | `k03_b200_hadamard_hadamard_transform` | 2 | `jit_kernel_hadamard_hadamard_transform` |
| 04 | `k04_b200_rope_apply_rope_inplace` | 3 | `jit_kernel_rope_apply_rope_inplace` |
| 05 | `k05_b200_dsv3_fused_a_gemm` | 4 | `sgl_kernel_dsv3_fused_a_gemm` |
| 06 | `k06_b200_fast_topk_transform_fused` | 5 | `sgl_kernel_fast_topk_transform_fused` |
| 07 | `k07_b200_fused_add_rmsnorm` | 6 | `sgl_kernel_fused_add_rmsnorm` |
| 08 | `k08_b200_rmsnorm` | 7 | `sgl_kernel_rmsnorm` |
| 09 | `k09_b200_quant_method_model_opt_fp4_linear_method_apply` | 0 | `sglang_quant_method_model_opt_fp4_linear_method_apply` |
| 10 | `k10_b200_quant_method_unquantized_linear_method_apply` | 1 | `sglang_quant_method_unquantized_linear_method_apply` |
| 11 | `k11_b200_attention_base_attn_backend_attention_backend_forward` | 2 | `srt_layers_attention_base_attn_backend_attention_backend_forward` |
| 12 | `k12_b200_layernorm_layernorm` | 3 | `srt_layers_layernorm_layernorm` |
| 13 | `k13_b200_moe_topk_fused_topk_deepseek` | 4 | `srt_layers_moe_topk_fused_topk_deepseek` |
| 14 | `k14_b200_quantization_fp4_utils_flashinfer_fp4_quantize_impl` | 5 | `srt_layers_quantization_fp4_utils_flashinfer_fp4_quantize_impl` |
| 15 | `k15_b200_quantization_modelopt_quant_fp4_gemm` | 6 | `srt_layers_quantization_modelopt_quant_fp4_gemm` |
| 16 | `k16_b200_deepseek_v2_flashinfer_dsv3_router_gemm` | 7 | `srt_models_deepseek_v2_flashinfer_dsv3_router_gemm` |

## Excluded: communication kernels (no launcher)

Communication kernels are not optimized here, so no launcher is generated for:

- `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce`
- `srt_distributed_parallel_state_inplace_all_reduce`
- `srt_distributed_parallel_state_outplace_all_reduce`
- `srt_distributed_parallel_state_reg_all_gather_into_tensor`
