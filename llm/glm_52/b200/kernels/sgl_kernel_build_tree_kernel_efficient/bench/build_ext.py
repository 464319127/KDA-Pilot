"""JIT-build helper for the standalone build_tree extension.

Baseline (baseline/build_tree_baseline.cu), candidate
(solution/build_tree_candidate.cu) and the pybind binding (bench/csrc/binding.cpp)
are compiled together in a SINGLE torch CUDA-extension build, so both sides share
the exact same compile flags, registration style, and Python call path (the
fairness requirement). The build is JIT and happens at import time, outside any
timed region.

Compile flags are symmetric and contain no one-sided fast-math.
"""

from __future__ import annotations

import os

from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # the kernel task folder

_EXT = None


def get_ext():
    """Build (once) and return the loaded extension module."""
    global _EXT
    if _EXT is None:
        _EXT = load(
            name="build_tree_ext",
            sources=[
                os.path.join(_ROOT, "baseline", "build_tree_baseline.cu"),
                os.path.join(_ROOT, "solution", "build_tree_candidate.cu"),
                os.path.join(_HERE, "csrc", "binding.cpp"),
            ],
            extra_include_paths=[os.path.join(_HERE, "csrc")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=True,
        )
    return _EXT
