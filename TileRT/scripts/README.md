# TileRT KDA launch scripts

One-click launchers that spin up a task-owned git worktree and start a Humanize
RLCR kernel-optimization loop for a TileRT DeepSeek-V3.2 fused kernel — the same
pattern as `diffusion/scripts/`, but tuned for TileRT (golden_forward oracle +
≥3× ncu reference + TileRT design levers; no upstream-SGLang copy step).

## Quick start

Launch the optimization loop for one kernel (run from the repo root):

```bash
TileRT/scripts/launch_kernels/b200_tilert_mla_decode.sh          # PureMlaDsv32  (53% of decode)
TileRT/scripts/launch_kernels/b200_tilert_fused_moe.sh           # FusedMoe      (37% of decode)
TileRT/scripts/launch_kernels/b200_tilert_sparse_select_mla.sh   # SparseSelectMla (8% of decode)
```

Each wrapper just calls the generic launcher with the right task dir. To dry-run
(create the worktree + draft, print the gen-plan/RLCR commands, no Claude):

```bash
KDA_NO_CLAUDE=1 TileRT/scripts/launch_kernels/b200_tilert_mla_decode.sh
```

Any other TileRT kernel task can be launched directly:

```bash
TileRT/scripts/launch_kda_kernel_task.sh 'TileRT/kernels/b200_tilert_head_proj_gemm'
```

## Why only these three have wrappers

The wrappers exist for the kernels whose **measured no-MTP decode CUDA share is
> 1%** (the `(NN%)` folders) — together ≈ 98% of decode time, so they are where
KDA effort pays off:

| wrapper | kernel | decode share |
|---|---|---|
| `b200_tilert_mla_decode.sh` | PureMlaDsv32 (MLA decode) | 52.8% |
| `b200_tilert_fused_moe.sh` | FusedMoe (MoE, FP4 experts) | 36.5% |
| `b200_tilert_sparse_select_mla.sh` | SparseSelectMlaDsv32 (GPU0 DSA-indexer MLA) | 7.6% |

Every other kernel is individually < 1% of decode (see `../KERNEL_REGISTRY.md`);
launch it directly with `launch_kda_kernel_task.sh` if needed.

## Notes

- The `(NN%)` decode-share suffix in the folder name is stripped from git
  branch/worktree names (wrappers set a clean `KDA_TASK_LABEL`).
- Env overrides (`KDA_BASE_BRANCH`, `KDA_MODEL`, `KDA_NO_CLAUDE`, …) are the same
  as the diffusion launcher; run `launch_kda_kernel_task.sh -h` for the full list.
- TileRT tasks are B200 (sm_100) only; the default remote host is the
  cirrascale-gpua83e B200 node.
