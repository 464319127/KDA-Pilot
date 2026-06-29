"""Build/load the LTX2 split-RoPE candidate CUDA module via tvm_ffi.cpp.load.

Compile flags are symmetric/conservative for bit-exactness (recorded in
docs/benchmark_method.md): -std=c++17 -O3 -gencode sm_100 -lineinfo, NO
--use_fast_math. The kernel uses explicit IEEE-rounded intrinsics, so codegen
cannot reorder the visible rounding. Links against torch (ATen) for the current
CUDA stream.
"""

import functools
from pathlib import Path

import torch
from torch.utils import cpp_extension
from tvm_ffi import cpp

_HERE = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def load_candidate_module():
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    torch_incs = [str(p) for p in cpp_extension.include_paths()]
    torch_libdirs = [str(p) for p in cpp_extension.library_paths()]

    extra_cflags = ["-std=c++17", "-O3", f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]
    extra_cuda_cflags = [
        "-std=c++17",
        "-O3",
        "-gencode=arch=compute_100,code=sm_100",
        "-lineinfo",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
    ]
    extra_ldflags = []
    for d in torch_libdirs:
        extra_ldflags += [f"-L{d}", f"-Wl,-rpath,{d}"]
    extra_ldflags += ["-lc10", "-lc10_cuda", "-ltorch_cpu", "-ltorch_cuda"]

    return cpp.load(
        name="ltx2_qknorm_split_rope_candidate",
        cuda_files=[str(_HERE / "kernel.cu")],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=torch_incs,
    )
