"""skinny_gemv config sweep vs cuBLAS on GLM-5.2 TP=8 decode shapes."""
import time

import torch

import sys
sys.path.insert(0, "/data/bbuf/repos/mini-sglang/python")
from minisgl.kernel.gemv import _jit_skinny_gemv_module

torch.manual_seed(0)
dev = "cuda"

SHAPES = [  # (K, N, name)
    (6144, 2112, "qkv_a"),
    (1536, 960, "q_b"),
    (640, 6144, "o_proj"),
    (6144, 512, "sh_gateup"),
    (256, 6144, "sh_down"),
    (12288, 6144, "eh_proj"),
    (6144, 18944, "lm_head"),
]


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


for M in (4, 1):
    print(f"===== M={M}", flush=True)
    for K, N, name in SHAPES:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w)
        t_ref = time_graph(lambda: torch.nn.functional.linear(x, w))
        best = None
        for rows in (1, 2, 4, 8, 16):
            for threads in (128, 256):
                if K % 8 != 0:
                    continue
                try:
                    mod = _jit_skinny_gemv_module(rows, threads)
                    mod.launch(x, w, out)
                    torch.cuda.synchronize()
                    rel = (ref.float() - out.float()).abs().max().item() / (
                        ref.float().abs().max().item() + 1e-9
                    )
                    if rel > 1e-2:
                        print(f"  {name} r{rows}t{threads}: WRONG rel={rel:.2e}", flush=True)
                        continue
                    t = time_graph(lambda: mod.launch(x, w, out))
                    if best is None or t < best[0]:
                        best = (t, rows, threads, rel)
                except Exception as e:
                    print(f"  {name} r{rows}t{threads}: ERR {str(e)[:80]}", flush=True)
        if best:
            t, rows, threads, rel = best
            verdict = "WIN" if t < t_ref * 0.95 else "lose"
            print(
                f"{name:10s} K={K:6d} N={N:6d} cublas={t_ref:6.2f}us best=r{rows}t{threads} {t:6.2f}us rel={rel:.1e} {verdict}",
                flush=True,
            )
