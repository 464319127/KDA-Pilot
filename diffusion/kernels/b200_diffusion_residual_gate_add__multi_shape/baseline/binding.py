"""Local baseline adapters exposed through the same destination-passing ABI as
the candidate.

``residual_gate_add``: the production-faithful baseline is SGLang's Triton fused
scale-shift kernel. ``out = residual + update * gate`` is served upstream by
``fuse_scale_shift_kernel(x=update, scale=gate, shift=residual,
scale_constant=0)`` -> ``update * (0 + gate) + residual``. SGLang PR #29361 adds
a native-CUDA fast path for exactly this pattern and benchmarks it against this
same Triton kernel as the reference, so the Triton kernel -- not the naive
two-launch eager ``mul``+``add`` -- is the right baseline. The kernel source is
vendored standalone in ``sglang_scale_shift_triton.py`` (see
``docs/baseline_source.md``).

``broadcast_add_4d``: the LTX 4D broadcast add is a single eager ``torch.add``
(one launch); there is no upstream Triton kernel for it.

No sglang import: the Triton kernel is vendored locally. Output tensors are
passed last and written in-place; nothing is allocated in the timed path.
"""

from __future__ import annotations

import torch

from .sglang_scale_shift_triton import fuse_scale_shift_kernel


def residual_gate_add(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor, out: torch.Tensor
) -> None:
    """out = residual + update * gate, via SGLang's Triton fuse_scale_shift_kernel.

    Maps x=update, scale=gate, shift=residual, scale_constant=0, so the kernel
    computes ``update * (0 + gate) + residual``. A [.,1,D] gate broadcasts over
    the sequence dim via a stride-0 expand. Writes into the preallocated ``out``
    (no allocation in the timed path).
    """
    fuse_scale_shift_kernel(update, gate, residual, scale_constant=0.0, out=out)


def broadcast_add_4d(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """out = a + b, with a [B,1,P,D] broadcasting over b's sequence dim."""
    torch.add(a, b, out=out)
