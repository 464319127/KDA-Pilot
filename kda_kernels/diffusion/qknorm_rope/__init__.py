"""kda_kernels.diffusion.qknorm_rope — CUDA-only KDA-optimized overlay.

This package contributes the following swap functions:

  - `sglang.jit_kernel.diffusion.qknorm_rope:fused_inplace_qknorm_rope`

Stub status: each function is either re-exported from SGLang
(`KDA_OPTIMIZED_<fn> = False`) or pulled from a promoted KDA impl
(`KDA_OPTIMIZED_<fn> = True`). Promotion is driven by
`scripts/export_kda_kernels/export.py <task-slug>`.
"""

from kda_kernels.diffusion.qknorm_rope.wrapper import fused_inplace_qknorm_rope  # noqa: F401

KDA_OPTIMIZED_fused_inplace_qknorm_rope = True

KDA_TASK_fused_inplace_qknorm_rope = 'b200_diffusion_qknorm_rope__multi_shape'
KDA_COMMIT_fused_inplace_qknorm_rope = '68e921ceeca306093b55e776ffd0cb256e0e90ae'
KDA_DATE_fused_inplace_qknorm_rope = '2026-05-31'
KDA_SPEEDUP_fused_inplace_qknorm_rope = '1.1089x'
