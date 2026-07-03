"""moe_decode correctness vs fp32 dequant reference + triton fused_experts, and
cuda-graph timing at M=1..8 on GLM-5.2 TP=8 decode shapes."""
import time

import torch

import sys
sys.path.insert(0, "/data/bbuf/repos/mini-sglang/python")
from minisgl.kernel.moe_decode import moe_decode

torch.manual_seed(0)
dev = "cuda"
E, H, I, K = 256, 6144, 256, 8

w13 = (torch.randn(E, 2 * I, H, device=dev) * 0.05).to(torch.float8_e4m3fn)
w2 = (torch.randn(E, H, I, device=dev) * 0.05).to(torch.float8_e4m3fn)
w13_s = torch.rand(E, 2 * I // 128, H // 128, device=dev) * 0.02 + 0.01
w2_s = torch.rand(E, H // 128, I // 128, device=dev) * 0.02 + 0.01


def ref_moe(x, ids, w):
    out = torch.zeros(x.shape[0], H, device=dev, dtype=torch.float32)
    s13 = torch.repeat_interleave(torch.repeat_interleave(w13_s, 128, 1), 128, 2)
    s2 = torch.repeat_interleave(torch.repeat_interleave(w2_s, 128, 1), 128, 2)
    for t in range(x.shape[0]):
        for s in range(K):
            e = int(ids[t, s])
            wg = w13[e].float() * s13[e]
            gu = x[t].float() @ wg.t()
            g, u = gu[:I], gu[I:]
            inter = torch.nn.functional.silu(g) * u
            wd = w2[e].float() * s2[e]
            out[t] += float(w[t, s]) * (inter @ wd.t())
    return out


def time_graph(fn, inner=20, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters / inner * 1e6


from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl

for M in (1, 4, 8):
    x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(E, device=dev)[:K] for _ in range(M)]
    ).to(torch.int32)
    w = torch.rand(M, K, device=dev, dtype=torch.float32) * 0.4

    ref = ref_moe(x, ids, w)
    mine = moe_decode(x, w13, w13_s, w2, w2_s, ids, w)
    tri = fused_experts_impl(
        x.contiguous(), w13, w2, w, ids, inplace=False,
        use_fp8_w8a8=True, w1_scale=w13_s, w2_scale=w2_s, block_shape=[128, 128],
    )
    denom = ref.abs().max().item() + 1e-9
    rel_mine = (mine.float() - ref).abs().max().item() / denom
    rel_tri = (tri.float() - ref).abs().max().item() / denom
    t_mine = time_graph(lambda: moe_decode(x, w13, w13_s, w2, w2_s, ids, w))
    t_tri = time_graph(lambda: fused_experts_impl(
        x.contiguous(), w13, w2, w, ids, inplace=False,
        use_fp8_w8a8=True, w1_scale=w13_s, w2_scale=w2_s, block_shape=[128, 128]))
    print(
        f"M={M}: rel_mine={rel_mine:.2e} rel_triton={rel_tri:.2e} | "
        f"mine={t_mine:6.2f}us triton={t_tri:6.2f}us",
        flush=True,
    )
