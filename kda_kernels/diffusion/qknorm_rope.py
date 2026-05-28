"""kda_kernels mirror of `sglang.jit_kernel.diffusion.qknorm_rope`.

Stub status: each function below is either re-exported from sglang
(KDA_OPTIMIZED_<fn>=False) or pulled from a promoted KDA impl
(KDA_OPTIMIZED_<fn>=True). See scripts/export_kda_kernels/export.py
for the swap rule.
"""

from sglang.jit_kernel.diffusion.qknorm_rope import fused_inplace_qknorm_rope  # noqa: F401

KDA_OPTIMIZED_fused_inplace_qknorm_rope = False

