#!/usr/bin/env python3
"""Build the task-local ABI `topk_transform_abi` for fast_topk_transform_fused.

Compiles the recovered baseline kernel, the workspace candidate, and the binding into ONE
extension module, so baseline and candidate are built with identical flags (symmetric by
construction — no one-sided fast-math, no asymmetric builder path). Run on the remote B200:

    REMOTE_GPU_ID=1 TORCH_CUDA_ARCH_LIST=10.0 python3 solution/build.py

The .so is emitted into solution/; bench/adapter.py adds solution/ to sys.path before
importing `topk_transform_abi`, so the benchmark's spawn subprocesses can import it.
"""
import sys
from pathlib import Path

from torch.utils.cpp_extension import load

SOLUTION_DIR = Path(__file__).resolve().parent
KERNEL_DIR = SOLUTION_DIR.parent
BASELINE_TOPK = KERNEL_DIR / "baseline" / "sgl-kernel" / "csrc" / "elementwise" / "topk.cu"

SOURCES = [
    str(BASELINE_TOPK),
    str(SOLUTION_DIR / "candidate_topk_transform.cu"),
    str(SOLUTION_DIR / "binding.cpp"),
]

# Symmetric flags for both sides (one module). Default: no fast-math (numerics-preserving).
EXTRA_CUDA_CFLAGS = ["-O3", "-lineinfo"]
EXTRA_CFLAGS = ["-O3"]


def build(verbose: bool = True):
    for src in SOURCES:
        if not Path(src).exists():
            raise FileNotFoundError(f"missing source: {src}")
    module = load(
        name="topk_transform_abi",
        sources=SOURCES,
        build_directory=str(SOLUTION_DIR),
        extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
        extra_cflags=EXTRA_CFLAGS,
        verbose=verbose,
    )
    return module


if __name__ == "__main__":
    build(verbose=True)
    print(f"built topk_transform_abi into {SOLUTION_DIR}", file=sys.stderr)
    print("OK")
