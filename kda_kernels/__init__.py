"""kda_kernels — KDA-optimized sglang diffusion kernels.

Usage from a fresh Python session::

    import kda_kernels
    results = kda_kernels.install()
    # results is a list of (sglang_path, kda_path, status) tuples.

Or activate transparently for every `import sglang` by applying
`patches/sglang_kda_kernels.patch` to your sglang checkout; that patch
adds a small `try: import kda_kernels; kda_kernels.install()` block at
the end of `sglang/__init__.py`.

Each function under `kda_kernels.diffusion.*` mirrors the matching
`sglang.jit_kernel.diffusion.*` function. Functions that have been
optimized via a KDA task carry a `KDA_OPTIMIZED_<fn> = True` flag;
functions still on the baseline keep the flag at False and install()
leaves them untouched.
"""

from kda_kernels._install import install, status, uninstall

__version__ = "0.1.0"
__all__ = ["install", "status", "uninstall", "__version__"]
