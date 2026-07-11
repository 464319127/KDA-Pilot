"""Diagnostic runner: the SERVING-BUILT module (sglang load_jit .so).

Loads the exact .so the flag-ON server used (built by sglang's load_jit into
the tvm-ffi cache) and exposes the same launch() ABI, so the harness can A/B
the serving build against the flashinfer original and against the
harness-built port. Diagnostic only — used to isolate the flag-ON serving
divergence (build-flavor vs integration-level).
"""

from __future__ import annotations

import functools
import glob
import os


@functools.cache
def _module():
    from tvm_ffi import load_module

    cands = sorted(
        glob.glob(os.path.expanduser(
            "~/.cache/tvm-ffi/sgl_kernel_jit_mnnvl_ar_fused_*/**/*.so"),
            recursive=True)
        + glob.glob(os.path.expanduser(
            "~/.cache/tvm-ffi/sgl_kernel_jit_mnnvl_ar_fused_*/*.so"))
    )
    if not cands:
        raise RuntimeError("serving-built mnnvl_ar_fused .so not found in tvm-ffi cache")
    path = cands[-1]
    print(f"[jit_serving_build] loading {path}")
    return load_module(path)


def name() -> str:
    return "jit-serving-build(load_jit-so)"


def launch(input, output, residual_in, residual_out, gamma, ws_rank,
           epsilon: float, launch_with_pdl: bool) -> None:
    _module().trtllm_mnnvl_allreduce_fusion(
        input, ws_rank.mc_ptr, ws_rank.uc_ptrs_dev, ws_rank.uc_ptr_local,
        ws_rank.buffer_flags, ws_rank.world_size, ws_rank.rank,
        True, launch_with_pdl, True,
        output, residual_out, residual_in, gamma, float(epsilon), 0.0,
    )
