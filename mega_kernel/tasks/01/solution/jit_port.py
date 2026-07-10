"""Candidate runner: the verbatim jit_kernel port of the MNNVL fused AR kernel.

Builds `solution/mnnvl_ar_fused/csrc/mnnvl_ar_fused.cu` with TVM-FFI using the
sglang jit_kernel default flag set (`-std=c++20 -O3 --expt-relaxed-constexpr`)
pinned to the sm_103a target — the same loader stack sglang's jit_kernel uses,
without importing the live sglang checkout (benchmark/correctness runtime must
not touch it). Exposes the same launch() ABI as baseline/fi_original.py.
"""

from __future__ import annotations

import functools
import os
import pathlib

_CSRC = pathlib.Path(__file__).resolve().parent / "mnnvl_ar_fused" / "csrc"
_ARCH = "10.3a"  # sm_103a per task contract
# --use_fast_math matches the float codegen of the DEPLOYED baseline binary
# (flashinfer-jit-cache 0.6.12+cu130 AOT wheel: FADD.FTZ throughout + MUFU.RCP
# division; verified by SASS inspection 2026-07-10). Without it, an IEEE build
# of the same verbatim source diverges bf16-bitwise on subnormal/zero-sign
# value classes that real serving activations contain (see docs/results.md,
# flag-ON divergence root cause). This is baseline-matching, not a one-sided
# fast-math advantage: the baseline binary carries the same flags.
_FLAGS_CUDA = ["-std=c++20", "-O3", "--expt-relaxed-constexpr", "--use_fast_math",
               "-DSGL_CUDA_ARCH=1030"]
_FLAGS_CPP = ["-std=c++20", "-O3"]


def _include_paths():
    # tvm_ffi_utils.h ships in flashinfer's data/csrc (same helper the baseline
    # binding uses); build-utility headers only — the kernel itself is the
    # workspace-owned verbatim port.
    import flashinfer

    data = pathlib.Path(flashinfer.__file__).resolve().parent / "data"
    return [str(_CSRC), str(data / "csrc"), str(data / "include")]


@functools.cache
def _module():
    from tvm_ffi.cpp import load

    old = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = _ARCH
    try:
        return load(
            "mnnvl_ar_fused_port_fm",  # _fm: fast-math build (cache key busts stale IEEE build)
            cuda_files=[str(_CSRC / "mnnvl_ar_fused.cu")],
            extra_cflags=list(_FLAGS_CPP),
            extra_cuda_cflags=list(_FLAGS_CUDA),
            extra_include_paths=_include_paths(),
        )
    finally:
        if old is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = old


def name() -> str:
    return "jit-port-mnnvl_ar_fused"


def build_info() -> dict:
    return {"arch": _ARCH, "cuda_flags": _FLAGS_CUDA, "cpp_flags": _FLAGS_CPP}


def launch(
    input,
    output,
    residual_in,
    residual_out,
    gamma,
    ws_rank,
    epsilon: float,
    launch_with_pdl: bool,
) -> None:
    """Identical ABI to baseline/fi_original.launch (oneshot fused path)."""
    _module().trtllm_mnnvl_allreduce_fusion(
        input,
        ws_rank.mc_ptr,
        ws_rank.uc_ptrs_dev,
        ws_rank.uc_ptr_local,
        ws_rank.buffer_flags,
        ws_rank.world_size,
        ws_rank.rank,
        True,  # rmsnorm_fusion
        launch_with_pdl,
        True,  # use_oneshot
        output,
        residual_out,
        residual_in,
        gamma,
        float(epsilon),
        0.0,
    )
