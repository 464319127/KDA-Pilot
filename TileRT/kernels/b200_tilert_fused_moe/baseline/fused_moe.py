"""Correct PyTorch MoE-decode baseline: sigmoid + noaux_tc group-limited routing + shared
expert. Mirrors DeepSeek-V3.2 / MiniTileRT dsv32_ref.moe_gate+ffn (validated structurally)."""
import torch, torch.nn.functional as F
def fused_moe_baseline(x, gate_w, e_bias, experts, shared, n_group=8, topk_group=4,
                       top_k=8, routed_scale=2.5, norm_topk=True):
    # x:[N,D]; gate_w:[E,D]; e_bias:[E]; experts: list of (gate,up,down) [.,D]/[D,.]; shared: (gate,up,down)
    scores = F.linear(x.float(), gate_w.float()).sigmoid()
    choice = scores + e_bias.float(); N = x.shape[0]
    grp = choice.view(N, n_group, -1)
    gidx = grp.topk(2, -1)[0].sum(-1).topk(topk_group, -1)[1]
    gmask = torch.zeros(N, n_group, dtype=torch.bool, device=x.device).scatter_(1, gidx, True)
    choice = choice.masked_fill(~gmask[:, :, None].expand_as(grp).reshape(N, -1), float("-inf"))
    idx = choice.topk(top_k, -1)[1]; wt = scores.gather(1, idx)
    if norm_topk: wt = wt / (wt.sum(-1, keepdim=True) + 1e-20)
    wt = wt * routed_scale
    out = torch.zeros_like(x)
    for s in range(N):
        for j in range(top_k):
            g, u, d = experts[int(idx[s, j])]
            out[s] += wt[s, j].to(x.dtype) * F.linear(F.silu(F.linear(x[s], g)) * F.linear(x[s], u), d)
    sg, su, sd = shared
    out = out + F.linear(F.silu(F.linear(x, sg)) * F.linear(x, su), sd)
    return out
