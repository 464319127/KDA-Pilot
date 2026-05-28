"""kda_kernels.diffusion.rotary_embedding — CUDA-only KDA-optimized overlay.

This package contributes the following swap functions:

  - `sglang.jit_kernel.diffusion.triton.rotary:apply_rotary_embedding`
  - `sglang.jit_kernel.diffusion.triton.ltx2_rotary:apply_ltx2_split_rotary_emb`

Stub status: each function is either re-exported from SGLang
(`KDA_OPTIMIZED_<fn> = False`) or pulled from a promoted KDA impl
(`KDA_OPTIMIZED_<fn> = True`). Promotion is driven by
`scripts/export_kda_kernels/export.py <task-slug>`, which copies
the task's CUDA `.cu` / `.cuh` sources + Python wrapper into this
directory and rewrites this `__init__.py` to import the wrapper.

After promotion the directory layout becomes::

    kda_kernels/diffusion/<family>/
        __init__.py     # this file (rewritten to import wrapper)
        wrapper.py      # Python wrapper that JIT-compiles the CUDA
        kernel.cu       # native CUDA source
        kernel.cuh      # CUDA headers
        KDA_STATUS.md   # task / commit / date / speedup stamps
"""

from sglang.jit_kernel.diffusion.triton.rotary import apply_rotary_embedding  # noqa: F401  (sglang baseline; replaced after export)
from sglang.jit_kernel.diffusion.triton.ltx2_rotary import apply_ltx2_split_rotary_emb  # noqa: F401  (sglang baseline; replaced after export)

KDA_OPTIMIZED_apply_rotary_embedding = False
KDA_OPTIMIZED_apply_ltx2_split_rotary_emb = False
