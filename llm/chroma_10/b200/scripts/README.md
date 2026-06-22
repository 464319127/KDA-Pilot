# chroma_10 B200 KDA kernel launchers

One-click KDA task launchers for the chroma_10 (`FlashLabs/Chroma-4B`) B200 kernel
optimization tasks under `llm/chroma_10/b200/kernels/`. Same flow and rules as
`llm/glm_52/b200/scripts/` (CUDA-only candidate, warp-spec profiling, per-shape
dispatch, round-robin GPU pin). The launcher draft mandates the shared contract
docs under `llm/docs/` before any implementation.

## Usage

```bash
# Launch one kernel task (creates a worktree + starts Claude Code):
llm/chroma_10/b200/scripts/launch_kernels/k01_b200_layers_activation_silu_and_mul.sh

# Prepare the worktree + draft without launching Claude (dry run):
KDA_NO_CLAUDE=1 llm/chroma_10/b200/scripts/launch_kernels/<kNN_...>.sh

# Launch any kernel folder directly:
llm/chroma_10/b200/scripts/launch_kda_kernel_task.sh llm/chroma_10/b200/kernels/<dir>
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
| 01 | `k01_b200_layers_activation_silu_and_mul` | 0 | `sglang_srt_layers_activation_silu_and_mul` |
| 02 | `k02_b200_layers_layernorm_rmsnorm` | 1 | `sglang_srt_layers_layernorm_rmsnorm` |
| 03 | `k03_b200_layers_linear_column_parallel_linear_forward` | 2 | `sglang_srt_layers_linear_column_parallel_linear_forward` |
| 04 | `k04_b200_layers_linear_row_parallel_linear_forward` | 3 | `sglang_srt_layers_linear_row_parallel_linear_forward` |
| 05 | `k05_b200_layers_quantization_unquant_unquantized_linear_method_apply` | 4 | `sglang_srt_layers_quantization_unquant_unquantized_linear_method_apply` |
| 06 | `k06_b200_layers_vocab_parallel_embedding_vocab_parallel_embedding_forward` | 5 | `sglang_srt_layers_vocab_parallel_embedding_vocab_parallel_embedding_forward` |

## Excluded: communication kernels (no launcher)

Communication kernels are not optimized here, so no launcher is generated for:

- (none)
