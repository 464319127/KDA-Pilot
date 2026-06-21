"""Correct PyTorch MLA-decode baseline (dense; == DSA for ctx<=2048). Mirrors the
MLA math in MiniTileRT dsv32_ref.mla_attention (validated structurally)."""
import math, torch
def mla_decode_baseline(q_nope, q_pe, k_nope, k_pe, v):
    # q_*: [S,H,*]; k_nope:[T,H,nope]; k_pe:[T,rope]; v:[T,H,vdim]
    H = q_nope.shape[1]
    qh = torch.cat([q_nope, q_pe], -1).float()
    kh = torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, H, -1)], -1).float()
    scale = 1.0 / math.sqrt(qh.shape[-1])
    attn = torch.einsum("shd,thd->hst", qh, kh) * scale
    S = qh.shape[0]
    attn = (attn + torch.triu(torch.full((S, S), float("-inf"), device=qh.device), 1)).softmax(-1)
    return torch.einsum("hst,thv->shv", attn, v.float())   # [S,H,vdim]
