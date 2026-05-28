"""kda_kernels mirror of `sglang.jit_kernel.diffusion.triton.norm`.

Stub status: until the matching KDA task under `kernels/` promotes its
optimized candidate via `scripts/export_kda_kernels/export.py`, every
function below re-exports the SGLang baseline and the
`KDA_OPTIMIZED_<fn>` flag stays False, so `kda_kernels.install()` is a
no-op for these functions.

After export, the import line for each promoted function is replaced
by an import from `kda_kernels._impls.<task-slug>.register`, and the
corresponding `KDA_OPTIMIZED_<fn>` flag flips to True. `install()`
then swaps that single function on the SGLang side at runtime; the
rest of `sglang.jit_kernel.diffusion.triton.norm` is untouched.
"""

from sglang.jit_kernel.diffusion.triton.norm import rms_norm_fn  # noqa: F401
from sglang.jit_kernel.diffusion.triton.norm import norm_infer  # noqa: F401

KDA_OPTIMIZED_rms_norm_fn = False
KDA_OPTIMIZED_norm_infer = False
