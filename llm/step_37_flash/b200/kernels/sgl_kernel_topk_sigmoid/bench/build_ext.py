"""Build + load helper for the standalone topk_sigmoid extension, exposing an
ABI-agnostic call surface (`baseline`, `candidate`, `noop`, `route`) so the rest of
the harness (adapter.py, correctness.py, floor_probe.py) does not depend on the
underlying binding mechanism.

ABI: local direct-symbol TVM-FFI (`TVM_FFI_DLL_EXPORT_TYPED_FUNC` /
`tvm::ffi::TensorView`, destination-passing, current CUDA stream). The recovered
baseline (baseline/topk_sigmoid_baseline.cu, vendored verbatim) and the native-CUDA
candidate binding (solution/topk_sigmoid_ext.cu, which includes the fused kernel
solution/topk_sigmoid_candidate.cuh) are compiled TOGETHER in ONE module with
SYMMETRIC flags (`-std=c++17 -O3` + native gencode, NO --use_fast_math), so both
sides share an identical registration/export/build style and call path. Torch
tensors + python ints are marshalled to TensorView / int64_t by tvm-ffi automatically.
"""

from __future__ import annotations

import functools
import pathlib

import torch

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent  # the kernel task folder
_BASELINE_CU = _ROOT / "baseline" / "topk_sigmoid_baseline.cu"
_EXT_CU = _ROOT / "solution" / "topk_sigmoid_ext.cu"
_INCLUDES = [_HERE / "csrc", _ROOT / "baseline", _ROOT / "solution"]


def _gencode_flag() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"


def candidate_compile_flags() -> dict:
    """Exact flag sets used to build BOTH sides (provenance record). Symmetric;
    no one-sided fast-math."""
    return {
        "extra_cflags": ["-std=c++17", "-O3"],
        "extra_cuda_cflags": ["-std=c++17", "-O3", _gencode_flag()],
        "ldlibs": ["-lc10", "-lc10_cuda", "-ltorch_cpu", "-ltorch_cuda"],
        "use_fast_math": False,
    }


@functools.lru_cache(maxsize=None)
def get_ext():
    """Build (once) and return the loaded TVM-FFI module exposing
    topk_sigmoid_baseline / topk_sigmoid_candidate / topk_sigmoid_candidate_route /
    topk_sigmoid_noop."""
    from torch.utils import cpp_extension as torch_cpp
    from tvm_ffi.cpp import load

    include_paths = list(torch_cpp.include_paths(device_type="cuda"))
    library_paths = list(torch_cpp.library_paths(device_type="cuda"))
    flags = candidate_compile_flags()
    ldflags = [f"-L{p}" for p in library_paths]
    ldflags += [f"-Wl,-rpath,{p}" for p in library_paths]
    ldflags += flags["ldlibs"]

    return load(
        "topk_sigmoid_ext",
        cuda_files=[str(_BASELINE_CU), str(_EXT_CU)],
        extra_cflags=flags["extra_cflags"],
        extra_cuda_cflags=flags["extra_cuda_cflags"],
        extra_ldflags=ldflags,
        extra_include_paths=include_paths + [str(p) for p in _INCLUDES],
        build_directory=str(_HERE / ".build"),
    )


# ---- ABI-agnostic call surface (identical signatures for both sides) ----
# args: topk_weights, topk_indices, gating_output, renormalize (int 0/1), correction_bias
#       -> None (topk_weights / topk_indices written in place)
def baseline(*args) -> None:
    get_ext().topk_sigmoid_baseline(*args)


def candidate(*args) -> None:
    get_ext().topk_sigmoid_candidate(*args)


def noop(gating_output) -> None:
    get_ext().topk_sigmoid_noop(gating_output)


# Dispatch-route diagnostic: 1 = candidate fast path, 0 = baseline fallback (no launch).
# Same 5-arg signature as baseline/candidate.
def route(*args) -> int:
    return int(get_ext().topk_sigmoid_candidate_route(*args))


# ---- No-bias variants (missing-bias fallback): 4-arg signature, no correction_bias ----
# args: topk_weights, topk_indices, gating_output, renormalize (int 0/1) -> None (in place)
def baseline_nobias(*args) -> None:
    get_ext().topk_sigmoid_baseline_nobias(*args)


def candidate_nobias(*args) -> None:
    get_ext().topk_sigmoid_candidate_nobias(*args)


def route_nobias(*args) -> int:
    return int(get_ext().topk_sigmoid_candidate_route_nobias(*args))
