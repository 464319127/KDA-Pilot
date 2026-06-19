"""Correct PyTorch baseline for TileRT head_proj (LM-head GEMM).

Validated vs the real TileRT op `tilert::head_proj_op` on B200: feeding the
swizzled weight, TileRT's output matches this baseline to rel ~5e-3 (bf16 GEMM).
"""
import torch


def head_proj_baseline(hidden: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """hidden [seq, K] bf16, W [V, K] bf16 (plain row-major) -> logits [seq, V] f32."""
    return hidden.float() @ W.float().T


def make_inputs(seq=1, K=7168, V=16160, device="cuda", dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    h = torch.randn(seq, K, device=device, dtype=dtype, generator=g) / (K ** 0.5)
    W = torch.randn(V, K, device=device, dtype=dtype, generator=g) / (K ** 0.5)
    return h, W
