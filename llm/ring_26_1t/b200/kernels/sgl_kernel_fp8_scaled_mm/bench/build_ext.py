"""Build + load helper for the standalone fp8_scaled_mm extension.

Compiles the recovered baseline (baseline/fp8_scaled_mm_baseline.cu, which
#includes the verbatim baseline/fp8_gemm_kernel.cu) and the native-CUDA
candidate (solution/fp8_scaled_mm_candidate.cu) TOGETHER in ONE TVM-FFI module
with SYMMETRIC flags. Exposes an ABI-agnostic call surface so adapter.py /
correctness.py do not depend on the binding mechanism.

ABI: local direct-symbol TVM-FFI (TVM_FFI_DLL_EXPORT_TYPED_FUNC /
tvm::ffi::TensorView), destination-passing (output pre-allocated, passed last),
launch on at::cuda::getCurrentCUDAStream(). Matches the glm_52 sibling tasks.

CUTLASS is a build-time dependency pinned to NVIDIA/cutlass@57e3cfb4 (the commit
sgl-kernel FetchContent's at sglang main@34dd9c28; see docs/baseline_source.md).
Point CUTLASS_DIR at a checkout of that commit, or let the default
(<task>/.deps/cutlass) be created by bench/setup_cutlass.sh on the remote box.
"""
from __future__ import annotations

import functools
import os
import pathlib

import torch

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent  # the kernel task folder
_BASELINE_CU = _ROOT / "baseline" / "fp8_scaled_mm_baseline.cu"
_CANDIDATE_CU = _ROOT / "solution" / "fp8_scaled_mm_candidate.cu"
_BASELINE_INCLUDE = _ROOT / "baseline"      # math.hpp, utils.h, cutlass_extensions/
_ABI_INCLUDE = _HERE / "csrc"               # fp8_scaled_mm_abi.h

CUTLASS_PIN = "57e3cfb47a2d9e0d46eb6335c3dc411498efa198"


def _cutlass_dir() -> pathlib.Path:
    d = os.environ.get("CUTLASS_DIR")
    if d:
        return pathlib.Path(d)
    return _ROOT / ".deps" / "cutlass"


def _gencode_flags() -> list[str]:
    """Gencode for the build device. B200 (sm_100, cc 10.0) needs the
    arch-accelerated 'a' target (compute_100a/sm_100a) so CUTLASS Blackwell
    tcgen05/TMA features compile."""
    major, minor = torch.cuda.get_device_capability()
    if major == 10:  # Blackwell datacenter (sm_100 / sm_103)
        tag = f"{major}{minor}a"
    elif (major, minor) == (12, 0):  # sm_120 (RTX 50xx) also needs 'a'
        tag = "120a"
    else:
        tag = f"{major}{minor}"
    return [f"-gencode=arch=compute_{tag},code=sm_{tag}"]


def candidate_compile_flags() -> dict:
    """Exact flag sets used to build BOTH sides (provenance). Symmetric; no
    one-sided fast-math. CUTLASS flags mirror sgl-kernel/CMakeLists.txt."""
    cutlass = _cutlass_dir()
    cutlass_includes = [
        str(cutlass / "include"),
        str(cutlass / "tools" / "util" / "include"),
    ]
    cuda_flags = [
        "-std=c++17", "-O3",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_TEST_LEVEL=0",
        "--threads=1",
        "-lineinfo",  # source-line attribution for ncu (no codegen/numeric effect)
    ] + _gencode_flags()
    return {
        "extra_cflags": ["-std=c++17", "-O3"],
        "extra_cuda_cflags": cuda_flags,
        "cutlass_includes": cutlass_includes,
        "ldlibs": ["-lc10", "-lc10_cuda", "-ltorch_cpu", "-ltorch_cuda"],
        "use_fast_math": False,
    }


@functools.lru_cache(maxsize=None)
def get_ext():
    from torch.utils import cpp_extension as torch_cpp
    from tvm_ffi.cpp import load

    cutlass = _cutlass_dir()
    if not (cutlass / "include" / "cutlass" / "cutlass.h").exists():
        raise RuntimeError(
            f"CUTLASS not found at {cutlass}. Set CUTLASS_DIR or run "
            f"bench/setup_cutlass.sh to fetch NVIDIA/cutlass@{CUTLASS_PIN}."
        )

    include_paths = list(torch_cpp.include_paths(device_type="cuda"))
    library_paths = list(torch_cpp.library_paths(device_type="cuda"))
    flags = candidate_compile_flags()
    ldflags = [f"-L{p}" for p in library_paths]
    ldflags += [f"-Wl,-rpath,{p}" for p in library_paths]
    ldflags += flags["ldlibs"]

    return load(
        "fp8_scaled_mm_ext",
        cuda_files=[str(_BASELINE_CU), str(_CANDIDATE_CU)],
        extra_cflags=flags["extra_cflags"],
        extra_cuda_cflags=flags["extra_cuda_cflags"],
        extra_ldflags=ldflags,
        extra_include_paths=include_paths
        + flags["cutlass_includes"]
        + [str(_BASELINE_INCLUDE), str(_ABI_INCLUDE)],
        build_directory=str(_HERE / ".build"),
    )


# ---- ABI-agnostic call surface (identical signatures for both sides) ----
# args: a, b, scale_a, scale_b, out  -> None (out written in place)
def baseline(*args) -> None:
    get_ext().fp8_scaled_mm_baseline(*args)


def candidate(*args) -> None:
    get_ext().fp8_scaled_mm_candidate(*args)


# Test-only bias-capable baseline (AC-3.1 bias edge): a, b, scale_a, scale_b, bias, out.
def baseline_bias(*args) -> None:
    get_ext().fp8_scaled_mm_baseline_bias(*args)


# Dispatch-route diagnostic: 1 = candidate fast path, 0 = baseline fallback (no launch).
def route(*args) -> int:
    return int(get_ext().fp8_scaled_mm_candidate_route(*args))
