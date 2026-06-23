#!/usr/bin/env python3
"""Build + load the task-local TVM-FFI ABI `topk_transform_abi` for fast_topk_transform_fused.

Compiles the recovered baseline kernel, the workspace candidate, and the TVM-FFI binding into
ONE module via tvm_ffi.cpp.load_inline, so baseline and candidate are built with identical flags
(symmetric by construction). The binding uses TVM_FFI_DLL_EXPORT_TYPED_FUNC + tvm::ffi::TensorView
(KDA standalone-benchmark ABI); the baseline is ATen-based, so torch include/lib paths are linked.

Run on the remote B200 (cwd = kernel folder):
    TORCH_CUDA_ARCH_LIST=10.0 python3 solution/build.py
bench/adapter.py imports load_abi() from this file, so the build is shared/cached.
"""
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension as tce
import tvm_ffi.cpp

SOLUTION_DIR = Path(__file__).resolve().parent
KERNEL_DIR = SOLUTION_DIR.parent
BASELINE_TOPK = KERNEL_DIR / "baseline" / "sgl-kernel" / "csrc" / "elementwise" / "topk.cu"
CANDIDATE = SOLUTION_DIR / "candidate_topk_transform.cu"
BINDING = SOLUTION_DIR / "binding.cu"

# B200 = compute capability 10.0 (sm_100). Symmetric flags for both sides; no fast-math.
CUDA_ARCH = "100"


def _sources():
    out = []
    for p in (BASELINE_TOPK, CANDIDATE, BINDING):
        if not p.exists():
            raise FileNotFoundError(f"missing source: {p}")
        out.append(p.read_text())
    return out


def _torch_paths():
    """torch include/lib paths, robust to the include_paths() signature change in torch 2.11."""
    for kwargs in ({"device_type": "cuda"}, {"cuda": True}, {}):
        try:
            return list(tce.include_paths(**kwargs)), list(tce.library_paths(**kwargs))
        except TypeError:
            continue
    return list(tce.include_paths()), list(tce.library_paths())


def load_abi():
    """Build (cached) and load the TVM-FFI module exposing baseline + candidate."""
    torch_inc, torch_lib = _torch_paths()
    ldflags = []
    for d in torch_lib:
        ldflags += [f"-L{d}", f"-Wl,-rpath,{d}"]
    ldflags += ["-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_cuda", "-lc10_cuda"]
    abi_flag = "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    return tvm_ffi.cpp.load_inline(
        name="topk_transform_abi",
        cuda_sources=_sources(),
        functions=["fast_topk_transform_fused_baseline", "fast_topk_transform_fused_candidate"],
        extra_include_paths=torch_inc,
        extra_cflags=["-O3", "-std=c++17", abi_flag],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "-lineinfo", abi_flag,
            f"-gencode=arch=compute_{CUDA_ARCH},code=sm_{CUDA_ARCH}",
        ],
        extra_ldflags=ldflags,
        build_directory=str(SOLUTION_DIR),
    )


if __name__ == "__main__":
    m = load_abi()
    fns = [f for f in dir(m) if "fast_topk" in f]
    print(f"built+loaded topk_transform_abi: {fns}", file=sys.stderr)
    print("OK")
