"""Correct PyTorch baseline for TileRT rmsnorm_quant. Matches tilert::rmsnorm_quant_op
output (rel 1.62e-3) and fp8 dequant (rel 2.37e-2) on B200."""
import torch
FP8_MAX = 448.0
def rmsnorm_quant_baseline(x, gamma, block=128, eps=1e-6):
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * gamma  # gamma f32
    hidden_out = y.to(torch.bfloat16)
    *lead, D = y.shape
    yb = y.reshape(*lead, D // block, block)
    amax = yb.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_MAX)
    q = (yb / scale).to(torch.float8_e4m3fn).reshape(*lead, D)
    return hidden_out, q, scale.squeeze(-1)
def make_inputs(seq=1, D=7168, device="cuda"):
    return (torch.randn(1, seq, D, device=device, dtype=torch.bfloat16),
            torch.randn(D, device=device, dtype=torch.float32))
