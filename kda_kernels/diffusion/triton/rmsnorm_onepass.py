"""kda_kernels mirror of `sglang.jit_kernel.diffusion.triton.rmsnorm_onepass`.

Stub status: until the matching KDA task under `kernels/` promotes its
optimized candidate via `scripts/export_kda_kernels/export.py`, every
function below re-exports the SGLang baseline and the
`KDA_OPTIMIZED_<fn>` flag stays False, so `kda_kernels.install()` is a
no-op for these functions.

After export, the import line for each promoted function is replaced
by an import from `kda_kernels._impls.<task-slug>.register`, and the
corresponding `KDA_OPTIMIZED_<fn>` flag flips to True. `install()`
then swaps that single function on the SGLang side at runtime; the
rest of `sglang.jit_kernel.diffusion.triton.rmsnorm_onepass` is untouched.
"""

from sglang.jit_kernel.diffusion.triton.rmsnorm_onepass import triton_one_pass_rms_norm  # noqa: F401

KDA_OPTIMIZED_triton_one_pass_rms_norm = False
