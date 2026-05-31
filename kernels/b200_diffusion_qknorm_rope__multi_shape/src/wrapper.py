"""Optimized fused in-place QK-Norm + RoPE wrapper (B200).

Exposes ``fused_inplace_qknorm_rope`` with the exact SGLang callsite contract.
Routes the production signature (bf16, contiguous, head_dim=128, rope_dim=128,
is_neox=False, equal Q/K heads, int32/int64 positions) to a workspace-owned
native CUDA kernel, and falls back to the SGLang baseline for everything else.

This module is what the promoted ``kda_kernels`` overlay imports
(``from kda_kernels.diffusion.qknorm_rope.wrapper import fused_inplace_qknorm_rope``),
so it uses no package-relative imports and locates its CUDA source by
``__file__``. The SGLang baseline is bound at import time, which keeps the
fallback non-recursive after ``kda_kernels.install()`` swaps the SGLang symbol
(install imports this module before the swap, capturing the original).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import torch

# Bind the SGLang baseline at import time (recursion-safe fallback).
try:
    from sglang.jit_kernel.diffusion.qknorm_rope import (
        fused_inplace_qknorm_rope as _SGLANG_BASELINE,
    )
except Exception:  # pragma: no cover - baseline only exists in the GPU env
    _SGLANG_BASELINE = None

_CSRC = Path(__file__).resolve().parent / "csrc" / "qknorm_rope_kernel.cu"

# Records the path taken by the most recent call ("cuda" | "fallback"), so
# tests can assert the production fast path actually ran (not inferred).
_LAST_DISPATCH: dict[str, Optional[str]] = {"path": None}


def last_dispatch_path() -> Optional[str]:
    return _LAST_DISPATCH["path"]


@functools.lru_cache(maxsize=1)
def _module():
    from torch.utils.cpp_extension import load

    return load(
        name="kda_qknorm_rope_b200",
        sources=[str(_CSRC)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )


def build() -> None:
    """Eagerly compile the CUDA extension (used to warm before benchmarking)."""
    _module()


def _supported(
    q, k, q_weight, k_weight, cos_sin_cache, positions, is_neox, head_dim, rope_dim
) -> bool:
    if is_neox:
        return False
    if not (q.is_cuda and k.is_cuda):
        return False
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16:
        return False
    if q.dim() != 3 or k.dim() != 3:
        return False
    if head_dim != 128 or rope_dim != 128:
        return False
    if q.size(-1) != 128 or k.size(-1) != 128:
        return False
    if q.size(0) != k.size(0):
        return False
    if q.size(1) != k.size(1):  # equal Q/K head counts
        return False
    if not (q.is_contiguous() and k.is_contiguous()):
        return False
    if q_weight.dtype is not torch.bfloat16 or k_weight.dtype is not torch.bfloat16:
        return False
    if q_weight.numel() != 128 or k_weight.numel() != 128:
        return False
    if not (q_weight.is_contiguous() and k_weight.is_contiguous()):
        return False
    if cos_sin_cache.dtype is not torch.float32 or not cos_sin_cache.is_contiguous():
        return False
    if cos_sin_cache.size(-1) != 128:
        return False
    if positions.dtype not in (torch.int32, torch.int64):
        return False
    if positions.dim() != 1 or positions.size(0) != q.size(0):
        return False
    if not positions.is_contiguous():
        return False
    if cos_sin_cache.dim() != 2:
        return False
    # All tensors must live on the same CUDA device as q (the kernel dereferences
    # them directly); otherwise route to the baseline rather than read bad memory.
    dev = q.device
    if (
        k.device != dev
        or q_weight.device != dev
        or k_weight.device != dev
        or cos_sin_cache.device != dev
        or positions.device != dev
    ):
        return False
    return True


def fused_inplace_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    eps: float = 1e-6,
    head_dim: int = 0,
    rope_dim: int = 0,
) -> None:
    resolved_head_dim = head_dim or q.size(-1)
    resolved_rope_dim = rope_dim or cos_sin_cache.size(-1)
    if _supported(
        q, k, q_weight, k_weight, cos_sin_cache, positions,
        is_neox, resolved_head_dim, resolved_rope_dim,
    ):
        _LAST_DISPATCH["path"] = "cuda"
        _module().fused_qknorm_rope(
            q, k, q_weight, k_weight, cos_sin_cache, positions,
            float(eps), bool(is_neox),
        )
        return None
    _LAST_DISPATCH["path"] = "fallback"
    # Guard against a pathological double-install where the bound baseline is this
    # very wrapper (would recurse forever); the normal install() flow binds the
    # original before swapping, so this never trips in practice.
    if _SGLANG_BASELINE is None or _SGLANG_BASELINE is fused_inplace_qknorm_rope:
        raise RuntimeError(
            "Unsupported signature and no non-recursive SGLang baseline available"
        )
    return _SGLANG_BASELINE(
        q, k, q_weight, k_weight, cos_sin_cache, positions,
        is_neox=is_neox, eps=eps, head_dim=head_dim, rope_dim=rope_dim,
    )
