"""kda_kernels mirror of `sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale`.

Stub status: until the matching KDA task under `kernels/` promotes its
optimized candidate via `scripts/export_kda_kernels/export.py`, every
function below re-exports the SGLang baseline and the
`KDA_OPTIMIZED_<fn>` flag stays False, so `kda_kernels.install()` is a
no-op for these functions.

After export, the import line for each promoted function is replaced
by an import from `kda_kernels._impls.<task-slug>.register`, and the
corresponding `KDA_OPTIMIZED_<fn>` flag flips to True. `install()`
then swaps that single function on the SGLang side at runtime; the
rest of `sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale` is untouched.
"""

from sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale import fused_norm_tanh_mul_add  # noqa: F401
from sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale import fused_norm_tanh_mul_add_norm_scale  # noqa: F401

KDA_OPTIMIZED_fused_norm_tanh_mul_add = False
KDA_OPTIMIZED_fused_norm_tanh_mul_add_norm_scale = False
