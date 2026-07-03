"""Per-kernel timing + config sweep for moe_decode."""
import time
import torch
import sys
sys.path.insert(0, "/data/bbuf/repos/mini-sglang/python")
from minisgl.kernel.moe_decode import _jit_moe_decode_module

torch.manual_seed(0)
dev = "cuda"
E, H, I, K = 256, 6144, 256, 8
w13 = (torch.randn(E, 2 * I, H, device=dev) * 0.05).to(torch.float8_e4m3fn).view(torch.uint8)
w2 = (torch.randn(E, H, I, device=dev) * 0.05).to(torch.float8_e4m3fn).view(torch.uint8)
w13_s = torch.rand(E, 2 * I // 128, H // 128, device=dev) * 0.02 + 0.01
w2_s = torch.rand(E, H // 128, I // 128, device=dev) * 0.02 + 0.01

def time_graph(fn, inner=20, iters=50):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner): fn()
    g.replay(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): g.replay()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters / inner * 1e6

for M in (1, 8):
    x = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, K, device=dev, dtype=torch.float32) * 0.4
    inter = torch.empty(M, K, I, dtype=torch.bfloat16, device=dev)
    out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
    for cfg in [(256,16,128,64),(512,32,128,64),(256,16,256,128),(256,16,128,128),(512,32,256,128)]:
        try:
            mod = _jit_moe_decode_module(*cfg)
            fn = lambda: mod.launch(x, w13, w13_s, w2, w2_s, ids, w, inter, out)
            fn(); torch.cuda.synchronize()
            t = time_graph(fn)
            # per-kernel split via profiler
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(20): fn()
                torch.cuda.synchronize()
            k1 = k2 = 0.0
            for ev in prof.key_averages():
                if "moe_g1" in ev.key: k1 = ev.device_time
                if "moe_g2" in ev.key: k2 = ev.device_time
            print(f"M={M} cfg={cfg}: total={t:6.2f}us g1={k1:6.2f} g2={k2:6.2f}", flush=True)
        except Exception as ex:
            print(f"M={M} cfg={cfg}: ERR {str(ex)[:100]}", flush=True)
