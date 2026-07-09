"""Candidate entry for `deep_gemm_fp8_fp8_bf16_nt` (GLM-5.2-FP8 / B200 task).

Single public dispatcher with the same destination-passing ABI as the baseline:

    run(a_fp8, a_scales_packed, b_fp8, b_scales_packed, out_bf16) -> None

Dispatch policy: host-side metadata predicates only (shape/dtype/stride checks —
no device syncs, no `.item()`); every layout not claimed by a native bucket
falls back to the recovered baseline (installed `deep_gemm.fp8_gemm_nt`), so
correctness is never lost on uncovered inputs. Any input transformation a
native bucket needs happens INSIDE this timed call.

Native buckets:
  decode_m1_gemv — the M == 1 decode regime (memory-bound GEMV), workspace-owned
  CUDA kernel in `solution/decode_gemv.cu`, JIT-built once for SM100 outside the
  timed region (the benchmark warms up before measuring). The exact supported
  layout is gated by `_supports_decode_m1`; everything else (M > 1, strided
  operands, wrong dtypes/scale layouts) falls back.
"""

import importlib.util
import os
from pathlib import Path

import torch

_SOLUTION_DIR = Path(__file__).resolve().parent
_TASK_DIR = _SOLUTION_DIR.parent


def _load_baseline_module():
    path = _TASK_DIR / "baseline" / "baseline_entry.py"
    spec = importlib.util.spec_from_file_location("task_baseline_entry_for_fallback", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fallback_run = _load_baseline_module().run

# Measurement-only flag, read ONCE at import (a per-call getenv would tax the
# ~20 us fallback path). When set, the M>1 diagnostic attempt kernel becomes
# reachable; default dispatch is behavior-identical to the promoted
# configuration (one extra short-circuited module-global boolean on the
# fallback branch; no getenv, no predicate call).
_DIAG_MGT1_ENABLED = os.environ.get("KDA_DIAG_NATIVE_MGT1") == "1"

_ext = None
_diag_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")  # B200 / SM100
        _ext = load(
            name="kda_glm52_decode_gemv",
            sources=[str(_SOLUTION_DIR / "decode_gemv.cu")],
            extra_cuda_cflags=[
                "-O3", "-lineinfo",
                # Explicit SM100a image: a preset TORCH_CUDA_ARCH_LIST without
                # 10.0a must not produce an extension that cannot launch on B200.
                "-gencode=arch=compute_100a,code=sm_100a",
            ],
            verbose=False,
        )
    return _ext


def _get_diag_ext():
    global _diag_ext
    if _diag_ext is None:
        from torch.utils.cpp_extension import load

        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")  # B200 / SM100
        _diag_ext = load(
            name="kda_glm52_mtile_gemm_diag",
            sources=[str(_SOLUTION_DIR / "mtile_gemm_diag.cu")],
            extra_cuda_cflags=[
                "-O3", "-lineinfo",
                # Explicit SM100a image: a preset TORCH_CUDA_ARCH_LIST without
                # 10.0a must not produce an extension that cannot launch on B200.
                "-gencode=arch=compute_100a,code=sm_100a",
            ],
            verbose=False,
        )
    return _diag_ext


def _supports_decode_m1(a_fp8, a_scales, b_fp8, b_scales, out_bf16):
    """True only for the EXACT layout `decode_gemv.cu` is proven to handle.

    The kernel indexes A/B/out as raw row-major bytes and reads packed UE8M0
    scales through their strides assuming the MN-major layout
    (int32, shape [rows, ceil(ceil(K/128)/4)], stride(0) == 1). Anything else
    must fall back to the baseline, which handles the general case.
    Metadata checks only — no host/device synchronization.
    """
    try:
        if a_fp8.dtype != torch.float8_e4m3fn or b_fp8.dtype != torch.float8_e4m3fn:
            return False
        if a_scales.dtype != torch.int32 or b_scales.dtype != torch.int32:
            return False
        if out_bf16.dtype != torch.bfloat16:
            return False
        if not (a_fp8.is_cuda and b_fp8.is_cuda and a_scales.is_cuda
                and b_scales.is_cuda and out_bf16.is_cuda):
            return False
        dev = a_fp8.device
        if not (b_fp8.device == dev and a_scales.device == dev
                and b_scales.device == dev and out_bf16.device == dev):
            return False
        if a_fp8.dim() != 2 or b_fp8.dim() != 2 or out_bf16.dim() != 2:
            return False
        if a_scales.dim() != 2 or b_scales.dim() != 2:
            return False
        m, k = a_fp8.shape
        n, kb = b_fp8.shape
        if m != 1 or kb != k or (k % 128) != 0 or (k % 16) != 0:
            return False
        if out_bf16.shape[0] != 1 or out_bf16.shape[1] != n:
            return False
        if not (a_fp8.is_contiguous() and b_fp8.is_contiguous() and out_bf16.is_contiguous()):
            return False
        words = ((k + 127) // 128 + 3) // 4
        if words > 32:
            # The kernel preloads packed scale words into registers and selects
            # them by warp shuffle (source lane == word index), which supports at
            # most 32 words (K <= 16384). Larger K falls back to the baseline.
            return False
        if tuple(a_scales.shape) != (1, words) or tuple(b_scales.shape) != (n, words):
            return False
        if a_scales.stride(0) != 1 or b_scales.stride(0) != 1:
            return False
        return True
    except (RuntimeError, AttributeError, IndexError):
        return False


def _supports_mtile_diag(a_fp8, a_scales, b_fp8, b_scales, out_bf16):
    """Exact gate for the M>1 DIAGNOSTIC attempt kernel (`mtile_gemm_diag.cu`).

    Covers the two attempted buckets only: the captured M == 113 small-prefill
    rows and tiny-K rows (K in {256, 512}) with M > 1. Same layout rigor as the
    decode predicate; metadata checks only, no host/device synchronization. This
    path is reachable only under KDA_DIAG_NATIVE_MGT1=1 (never promoted).
    """
    try:
        if a_fp8.dtype != torch.float8_e4m3fn or b_fp8.dtype != torch.float8_e4m3fn:
            return False
        if a_scales.dtype != torch.int32 or b_scales.dtype != torch.int32:
            return False
        if out_bf16.dtype != torch.bfloat16:
            return False
        if not (a_fp8.is_cuda and b_fp8.is_cuda and a_scales.is_cuda
                and b_scales.is_cuda and out_bf16.is_cuda):
            return False
        dev = a_fp8.device
        if not (b_fp8.device == dev and a_scales.device == dev
                and b_scales.device == dev and out_bf16.device == dev):
            return False
        if a_fp8.dim() != 2 or b_fp8.dim() != 2 or out_bf16.dim() != 2:
            return False
        if a_scales.dim() != 2 or b_scales.dim() != 2:
            return False
        m, k = a_fp8.shape
        n, kb = b_fp8.shape
        if not (m == 113 or (m > 1 and k in (256, 512))):
            return False
        if kb != k or (k % 128) != 0:
            return False
        if out_bf16.shape[0] != m or out_bf16.shape[1] != n:
            return False
        if not (a_fp8.is_contiguous() and b_fp8.is_contiguous() and out_bf16.is_contiguous()):
            return False
        words = ((k + 127) // 128 + 3) // 4
        if words > 32:
            # Same warp-shuffle scale-word preload limit as the decode kernel.
            return False
        if tuple(a_scales.shape) != (m, words) or tuple(b_scales.shape) != (n, words):
            return False
        if a_scales.stride(0) != 1 or b_scales.stride(0) != 1:
            return False
        return True
    except (RuntimeError, AttributeError, IndexError):
        return False


# bucket name -> fn(a_fp8, a_scales, b_fp8, b_scales, out_bf16) -> bool
# (the diagnostic predicate is exposed for negative-dispatch testing even though
# its route is only reachable under the measurement flag)
native_dispatch_predicates = {
    "decode_m1_gemv": _supports_decode_m1,
    "mtile_gemm_diag": _supports_mtile_diag,
}


def run(a_fp8, a_scales, b_fp8, b_scales, out_bf16):
    # Cheap M gate first: only decode-shaped calls pay the full layout predicate.
    # The predicate costs ~2 µs of attribute reads in eager mode, which is a real
    # tax on ~20 µs fallback calls; M>1 traffic must reach the baseline with one
    # metadata read. Unsupported M==1 layouts still go through the full predicate
    # and decline to the fallback (never into the native kernel).
    if a_fp8.size(0) == 1 and _supports_decode_m1(a_fp8, a_scales, b_fp8, b_scales, out_bf16):
        _get_ext().decode_m1_gemv(a_fp8, a_scales, b_fp8, b_scales, out_bf16)
    elif _DIAG_MGT1_ENABLED and _supports_mtile_diag(a_fp8, a_scales, b_fp8, b_scales, out_bf16):
        _get_diag_ext().mtile_gemm_diag(a_fp8, a_scales, b_fp8, b_scales, out_bf16)
    else:
        _fallback_run(a_fp8, a_scales, b_fp8, b_scales, out_bf16)


candidate = run
