# GLM-5.2 B200 KDA kernel launchers

One-click KDA task launchers for the GLM-5.2 (`zai-org/GLM-5.2-FP8`) B200 kernel
optimization tasks under `llm/glm_52/b200/kernels/`. Same flow as
`diffusion/scripts/`: each launcher creates a task-owned git worktree + RLCR
review-base branch, enters the kernel folder, bootstraps a Humanize gen-plan
draft from that kernel's `prompt.md`, and launches Claude Code scoped to the
task directory.

## Usage

```bash
# Launch one kernel task (creates a worktree and starts Claude Code):
llm/glm_52/b200/scripts/launch_kernels/k01_b200_rmsnorm.sh

# Prepare the worktree + draft without launching Claude (dry run):
KDA_NO_CLAUDE=1 llm/glm_52/b200/scripts/launch_kernels/k01_b200_rmsnorm.sh

# Launch any kernel folder directly:
llm/glm_52/b200/scripts/launch_kda_kernel_task.sh llm/glm_52/b200/kernels/<dir>
```

Run `launch_kda_kernel_task.sh --help` for all environment overrides
(`CLAUDE_MODEL`, `CLAUDE_EFFORT`, `KDA_BASE_BRANCH`, `KDA_RUN_ID`, ...).

Inside Claude Code the launcher prints the two commands to run:

```
/humanize:gen-plan --input .humanize/kernel-agent/draft.md --output .humanize/kernel-agent/refined-plan.md --direct
/humanize:start-rlcr-loop .humanize/kernel-agent/refined-plan.md --skip-quiz --claude-answer-codex --max 12 ... --base-branch <review-base>
```

## GPU assignment (round-robin 0-7)

Each wrapper pins its task to a B200 GPU, cycling `(N-1) % 8` so the 16 kernels
spread across the 8 cards (two kernels per GPU):

```
k01->0  k02->1  k03->2  k04->3  k05->4  k06->5  k07->6  k08->7
k09->0  k10->1  k11->2  k12->3  k13->4  k14->5  k15->6  k16->7
```

The wrapper exports `KDA_GPU_ID`; the launcher pins the task to that GPU and
tells the agent to `export REMOTE_GPU_ID=<id>` and use exactly that card for all
baseline/candidate/benchmark/profiler/NCU commands (after verifying it is idle).
Override per run with `KDA_GPU_ID=<id> .../kNN_*.sh`; launching
`launch_kda_kernel_task.sh` directly without `KDA_GPU_ID` falls back to
auto-selecting an idle GPU.

## Extra rules baked into every task draft

`launch_kda_kernel_task.sh` injects these into each kernel's gen-plan draft (on
top of the kernel's own `prompt.md`), so every launched task carries them:

1. **CUDA only** — iterate and implement the candidate in native CUDA. The
   promoted candidate must be a CUDA kernel built from workspace-owned C++/CUDA
   source (`.cu`/`.cuh`/`.cpp`/`.h`, nvcc or equivalent extension build; CUTLASS
   / CuTe C++ templates allowed). **No** Triton, TileLang, CuTe-DSL (Python
   `cute.compile`), `torch.compile`, or other DSL / prebuilt op as the candidate
   execution path — those may only be studied/ported into workspace CUDA. Python
   is for harnesses/bindings/benchmark/dispatch glue only.
2. **warp-specialization-report-skill** — when a candidate is a warp-specialized
   CUDA C++ kernel (CUTLASS/CuTe), profile it with
   `external/warp-specialization-report-skill` as a predict → stamp `clock()`
   timeline → reconcile loop (find stalls, confirm producer/consumer deps,
   verify warp overlap). Not applicable to Triton kernels (use ncu-report-skill
   there).
3. **Per-shape kernel dispatch** — the captured interface spans many shapes;
   write more than one specialized kernel when one config can't win across all
   regimes and dispatch at runtime by input shape/dtype/contiguity, with a
   cheap dispatch path and a baseline fallback for any uncovered shape.

## Kernels with launchers (16 compute kernels)

| # | launcher | kernel task dir |
|---|---|---|
| 01 | `k01_b200_rmsnorm` | `sgl_kernel_rmsnorm` |
| 02 | `k02_b200_fused_add_rmsnorm` | `sgl_kernel_fused_add_rmsnorm` |
| 03 | `k03_b200_layernorm` | `srt_layers_layernorm_layernorm` |
| 04 | `k04_b200_activation_inplace` | `jit_kernel_activation_run_activation_inplace` |
| 05 | `k05_b200_rope_inplace` | `jit_kernel_rope_apply_rope_inplace` |
| 06 | `k06_b200_per_token_group_quant_8bit_v2` | `jit_kernel_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2` |
| 07 | `k07_b200_per_token_group_quant_8bit_v2_custom_op` | `jit_kernel_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2_custom_op` |
| 08 | `k08_b200_hadamard_transform` | `jit_kernel_hadamard_hadamard_transform` |
| 09 | `k09_b200_grouped_topk` | `jit_kernel_grouped_topk_jit_grouped_topk_op` |
| 10 | `k10_b200_fast_topk_transform_fused` | `sgl_kernel_fast_topk_transform_fused` |
| 11 | `k11_b200_fused_store_index_k_cache` | `jit_kernel_fused_store_index_cache_fused_store_index_k_cache` |
| 12 | `k12_b200_attention_backend_forward` | `srt_layers_attention_base_attn_backend_attention_backend_forward` |
| 13 | `k13_b200_fp8_linear_method_apply` | `sglang_quant_method_fp8_linear_method_apply` |
| 14 | `k14_b200_deep_gemm_fp8_fp8_bf16_nt` | `srt_layers_quantization_fp8_kernel_deep_gemm_fp8_fp8_bf16_nt` |
| 15 | `k15_b200_build_tree_kernel_efficient` | `sgl_kernel_build_tree_kernel_efficient` |
| 16 | `k16_b200_verify_tree_greedy` | `sgl_kernel_verify_tree_greedy` |

## Excluded: communication kernels (no launcher)

Per the task owner, communication kernels are **not** optimized here, so no
launcher is generated for:

- `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce`
- `srt_distributed_parallel_state_inplace_all_reduce`
- `srt_distributed_parallel_state_outplace_all_reduce`
- `srt_distributed_parallel_state_reg_all_gather_into_tensor`
- `srt_distributed_parallel_state_reg_reduce_scatter_tensor`
