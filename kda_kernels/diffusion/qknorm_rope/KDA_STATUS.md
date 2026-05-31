# kda_kernels promotion status — qknorm_rope

| Field | Value |
|---|---|
| Task slug | `b200_diffusion_qknorm_rope__multi_shape` |
| Kernel-source commit | `68e921ceeca306093b55e776ffd0cb256e0e90ae` |
| Promotion date | 2026-05-31 |
| Reported geomean speedup | 1.1113x (all-shape) |
| Promoted functions | fused_inplace_qknorm_rope |

## Files

- `csrc`
- `register.py`
- `wrapper.py`

## Provenance note

- **Kernel source**: the promoted `.cu`/`wrapper` are the v3 kernel introduced in
  commit `68e921cee`; that matches `kernels/b200_diffusion_qknorm_rope__multi_shape/benchmark.csv`'s
  `kp_commit` column and the candidate source hash `qknorm_rope_kernel.cu@021eb444`.
- **Speedup evidence**: `1.1113x` is the all-shape geomean from the Round 2
  regenerated `benchmark.csv` (one invocation per CUDA-event sample, pristine Q/K
  reset per sample, GPU idle before=0% / after-idle=0%). The v3 kernel is
  unchanged across rounds 1–3, so the source commit and the benchmark evidence
  refer to the same kernel.
