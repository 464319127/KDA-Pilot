"""Self-contained TVM-FFI JIT build for the recovered baseline and the candidate.

Mirrors the upstream header-only ``load_inline`` path (the same mechanism SGLang's
``sglang.jit_kernel.utils.load_jit`` uses) but points at the workspace-owned copied
source under ``baseline/`` and ``solution/`` instead of importing a live SGLang install.
Baseline and candidate are built with identical flags, identical include paths, and the
identical exported symbol so the only measured difference is the kernel body.

Exported symbol: ``moe_fused_gate`` wrapping ``MoEFusedGateKernel::run`` (the same entry
point the upstream wrapper loads via
``load_jit("moe_fused_gate", cuda_files=["moe/moe_fused_gate.cuh"],
           cuda_wrappers=[("moe_fused_gate", "MoEFusedGateKernel::run")])``).
"""

from __future__ import annotations

import os
import pathlib

import torch
import tvm_ffi
from tvm_ffi.cpp import load_inline

_HERE = pathlib.Path(__file__).resolve().parent
TASK_ROOT = _HERE.parent

# Copied upstream headers (shared by both sides for build parity) live here.
SHARED_INCLUDE = TASK_ROOT / "baseline" / "include"
BASELINE_CUH = TASK_ROOT / "baseline" / "csrc" / "moe" / "moe_fused_gate.cuh"
CANDIDATE_CUH = TASK_ROOT / "solution" / "csrc" / "moe" / "moe_fused_gate_candidate.cuh"

# Exported typed function name + the C++ callable it wraps (same for both sides).
EXPORT_NAME = "moe_fused_gate"
KERNEL_ENTRY = "MoEFusedGateKernel::run"


def _tvm_ffi_include() -> list[str]:
    inc = pathlib.Path(tvm_ffi.__file__).resolve().parent / "include"
    return [str(inc)] if inc.is_dir() else []


def _device_arch() -> tuple[int, int]:
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major, minor


def _build(name: str, cuh_path: pathlib.Path, build_subdir: str):
    if not cuh_path.is_file():
        raise FileNotFoundError(f"kernel source not found: {cuh_path}")
    major, minor = _device_arch()
    arch = major * 100 + minor * 10  # e.g. B200 sm_100 -> 1000
    # Match the upstream JIT compile context.
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    build_dir = TASK_ROOT / ".jit_build" / build_subdir
    build_dir.mkdir(parents=True, exist_ok=True)
    cuda_sources = [
        f'#include "{cuh_path.resolve()}"',
        f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({EXPORT_NAME}, ({KERNEL_ENTRY}));",
    ]
    return load_inline(
        name,
        cpp_sources=[],
        cuda_sources=cuda_sources,
        extra_cflags=["-std=c++20", "-O3"],
        extra_cuda_cflags=[
            f"-DSGL_CUDA_ARCH={arch}",
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
        ],
        extra_ldflags=[],
        extra_include_paths=[str(SHARED_INCLUDE)] + _tvm_ffi_include(),
        build_directory=str(build_dir),
    )


_baseline_mod = None
_candidate_mod = None


def baseline_module():
    """Build (once) and return the recovered SGLang baseline module."""
    global _baseline_mod
    if _baseline_mod is None:
        _baseline_mod = _build("k05_baseline_moe_fused_gate", BASELINE_CUH, "baseline")
    return _baseline_mod


def candidate_module():
    """Build (once) and return the workspace-owned native-CUDA candidate module."""
    global _candidate_mod
    if _candidate_mod is None:
        _candidate_mod = _build("k05_candidate_moe_fused_gate", CANDIDATE_CUH, "candidate")
    return _candidate_mod


def has_candidate() -> bool:
    return CANDIDATE_CUH.is_file()
