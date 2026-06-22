# deepseek_v4 B200 KDA kernel launchers

One-click KDA task launchers for the deepseek_v4 (`deepseek-ai/DeepSeek-V4-Flash`) B200 kernel
optimization tasks under `llm/deepseek_v4/b200/kernels/`. Same flow and rules as
`llm/glm_52/b200/scripts/` (CUDA-only candidate, warp-spec profiling, per-shape
dispatch, round-robin GPU pin). The launcher draft mandates the shared contract
docs under `llm/docs/` before any implementation.

## Usage

```bash
# Launch one kernel task (creates a worktree + starts Claude Code):
llm/deepseek_v4/b200/scripts/launch_kernels/k01_b200_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2.sh

# Prepare the worktree + draft without launching Claude (dry run):
KDA_NO_CLAUDE=1 llm/deepseek_v4/b200/scripts/launch_kernels/<kNN_...>.sh

# Launch any kernel folder directly:
llm/deepseek_v4/b200/scripts/launch_kda_kernel_task.sh llm/deepseek_v4/b200/kernels/<dir>
```

## GPU assignment (round-robin 0-7)

Each wrapper pins its task to a B200 GPU, cycling `(N-1) % 8` so the
kernels spread across the 8 cards. Override per run with `KDA_GPU_ID=<id>`.

```
k01->0  k02->1  k03->2  k04->3  k05->4  k06->5
```

## Kernels with launchers (6 compute kernels)

| # | launcher | GPU | kernel task dir |
|---|---|---|---|
| 01 | `k01_b200_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2` | 0 | `jit_kernel_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2` |
| 02 | `k02_b200_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2_custom_op` | 1 | `jit_kernel_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2_custom_op` |
| 03 | `k03_b200_rmsnorm` | 2 | `sgl_kernel_rmsnorm` |
| 04 | `k04_b200_quant_method_fp8_linear_method_apply` | 3 | `sglang_quant_method_fp8_linear_method_apply` |
| 05 | `k05_b200_quant_method_unquantized_linear_method_apply` | 4 | `sglang_quant_method_unquantized_linear_method_apply` |
| 06 | `k06_b200_quantization_fp8_kernel_deep_gemm_fp8_fp8_bf16_nt` | 5 | `srt_layers_quantization_fp8_kernel_deep_gemm_fp8_fp8_bf16_nt` |

## Excluded: communication kernels (no launcher)

Communication kernels are not optimized here, so no launcher is generated for:

- `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce`
- `srt_distributed_parallel_state_inplace_all_reduce`
- `srt_distributed_parallel_state_outplace_all_reduce`
- `srt_distributed_parallel_state_reg_all_gather_into_tensor`
