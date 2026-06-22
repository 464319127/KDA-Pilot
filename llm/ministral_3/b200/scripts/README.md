# ministral_3 B200 KDA kernel launchers

One-click KDA task launchers for the ministral_3 (`mistralai/Ministral-3-14B-Instruct-2512`) B200 kernel
optimization tasks under `llm/ministral_3/b200/kernels/`. Same flow and rules as
`llm/glm_52/b200/scripts/` (CUDA-only candidate, warp-spec profiling, per-shape
dispatch, round-robin GPU pin). The launcher draft mandates the shared contract
docs under `llm/docs/` before any implementation.

## Usage

```bash
# Launch one kernel task (creates a worktree + starts Claude Code):
llm/ministral_3/b200/scripts/launch_kernels/k01_b200_activation_run_activation_inplace.sh

# Prepare the worktree + draft without launching Claude (dry run):
KDA_NO_CLAUDE=1 llm/ministral_3/b200/scripts/launch_kernels/<kNN_...>.sh

# Launch any kernel folder directly:
llm/ministral_3/b200/scripts/launch_kda_kernel_task.sh llm/ministral_3/b200/kernels/<dir>
```

## GPU assignment (round-robin 0-7)

Each wrapper pins its task to a B200 GPU, cycling `(N-1) % 8` so the
kernels spread across the 8 cards. Override per run with `KDA_GPU_ID=<id>`.

```
k01->0  k02->1  k03->2  k04->3  k05->4  k06->5  k07->6  k08->7
```

## Kernels with launchers (8 compute kernels)

| # | launcher | GPU | kernel task dir |
|---|---|---|---|
| 01 | `k01_b200_activation_run_activation_inplace` | 0 | `jit_kernel_activation_run_activation_inplace` |
| 02 | `k02_b200_kvcache_store_cache` | 1 | `jit_kernel_kvcache_store_cache` |
| 03 | `k03_b200_rope_apply_rope_inplace` | 2 | `jit_kernel_rope_apply_rope_inplace` |
| 04 | `k04_b200_fp8_scaled_mm` | 3 | `sgl_kernel_fp8_scaled_mm` |
| 05 | `k05_b200_fused_add_rmsnorm` | 4 | `sgl_kernel_fused_add_rmsnorm` |
| 06 | `k06_b200_rmsnorm` | 5 | `sgl_kernel_rmsnorm` |
| 07 | `k07_b200_quant_method_fp8_linear_method_apply` | 6 | `sglang_quant_method_fp8_linear_method_apply` |
| 08 | `k08_b200_attention_base_attn_backend_attention_backend_forward` | 7 | `srt_layers_attention_base_attn_backend_attention_backend_forward` |

## Excluded: communication kernels (no launcher)

Communication kernels are not optimized here, so no launcher is generated for:

- (none)
