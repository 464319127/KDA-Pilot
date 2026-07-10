"""03 AR+add+rmsnorm prototype: single-process 8-GPU bench (B300, sm103).

Correctness vs fp32 reference, then timing: per-device CUDA graphs (R rounds
captured per graph), all 8 replayed back-to-back, wall time / rounds.
Reference latency: flashinfer mnnvl oneshot fused AR+norm in serving = 8.7 us.
"""
import os, time, torch
from torch.utils.cpp_extension import load

WORLD = int(os.environ.get("AR_WORLD", "8"))
T, H = 6, 6144
EPS = 1e-6
ROUNDS_IN_GRAPH = 50

ext = load(name="ar_oneshot", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "ar_oneshot.cu")],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a"], verbose=False)

devs = list(range(WORLD))
import ctypes
_rt = ctypes.CDLL("libcudart.so")
for i in devs:
    torch.cuda.set_device(i)
    torch.zeros(1, device=f"cuda:{i}")  # init context
    for j in devs:
        if i != j:
            rc = _rt.cudaDeviceEnablePeerAccess(ctypes.c_int(j), ctypes.c_uint(0))
            if rc not in (0, 704):  # 704 = already enabled
                raise RuntimeError(f"peer access {i}->{j} failed rc={rc}")

# workspaces
slots = {}   # (rank) -> [world] tensors on device rank
flags = {}
epochs = {}
xs, residuals, gammas, outs, res_outs = {}, {}, {}, {}, {}
torch.manual_seed(0)
gamma_cpu = torch.randn(H, dtype=torch.bfloat16)
for r in devs:
    d = f"cuda:{r}"
    slots[r] = torch.zeros(WORLD, T * H, dtype=torch.bfloat16, device=d)
    flags[r] = torch.zeros(WORLD, dtype=torch.int32, device=d)
    epochs[r] = torch.zeros(1, dtype=torch.int32, device=d)
    xs[r] = torch.randn(T * H, dtype=torch.bfloat16, device=d)
    residuals[r] = torch.randn(T * H, dtype=torch.bfloat16, device=d) if r == 0 else torch.randn(T * H, dtype=torch.bfloat16, device=d)
    gammas[r] = gamma_cpu.to(d)
    outs[r] = torch.zeros(T * H, dtype=torch.bfloat16, device=d)
    res_outs[r] = torch.zeros(T * H, dtype=torch.bfloat16, device=d)

# same residual content on every rank (as in TP: residual is replicated)
for r in devs[1:]:
    residuals[r].copy_(residuals[0])

def ptr_table(r):
    # peer_slot_mine[p] = &slots[p][r, :]; peer_flag_mine[p] = &flags[p][r]
    ps = torch.tensor([slots[p][r].data_ptr() for p in devs], dtype=torch.int64, device=f"cuda:{r}")
    pf = torch.tensor([flags[p].data_ptr() + 4 * r for p in devs], dtype=torch.int64, device=f"cuda:{r}")
    ms = torch.tensor([slots[r][p].data_ptr() for p in devs], dtype=torch.int64, device=f"cuda:{r}")
    return ps, pf, ms

tables = {r: ptr_table(r) for r in devs}
GRID_X = 24

def launch(r):
    torch.cuda.set_device(r)
    ps, pf, ms = tables[r]
    ext.ar_oneshot(xs[r], residuals[r], gammas[r], outs[r], res_outs[r],
                   ps, pf, ms, flags[r], epochs[r], WORLD, T, H, EPS, GRID_X)

# ---- correctness (eager, one round) ----
for r in devs:
    launch(r)
for r in devs:
    torch.cuda.synchronize(r)
acc = residuals[0].float().cpu()
for r in devs:
    acc = acc + xs[r].float().cpu()
ref_res = acc.clone()
v = acc.view(T, H)
rms = torch.rsqrt((v * v).mean(-1, keepdim=True) + EPS)
ref = (v * rms * gamma_cpu.float()).flatten()
for r in devs:
    got = outs[r].float().cpu()
    rel = (got - ref).abs().max().item() / ref.abs().max().item()
    got_res = res_outs[r].float().cpu()
    rel_res = (got_res - ref_res).abs().max().item() / ref_res.abs().max().item()
    ok = rel < 2e-2 and rel_res < 2e-2
    print(f"rank{r}: out rel {rel:.2e} res rel {rel_res:.2e} {'OK' if ok else 'FAIL'}")

# ---- timing: per-device graphs with ROUNDS_IN_GRAPH rounds ----
graphs = {}
for r in devs:
    torch.cuda.set_device(r)
    for _ in range(3):
        launch(r)
for r in devs:
    torch.cuda.synchronize(r)
streams = {}
for r in devs:
    torch.cuda.set_device(r)
    streams[r] = torch.cuda.Stream(device=r)
    g = torch.cuda.CUDAGraph()
    ps, pf, ms = tables[r]
    with torch.cuda.graph(g, stream=streams[r]):
        for _ in range(ROUNDS_IN_GRAPH):
            ext.ar_oneshot(xs[r], residuals[r], gammas[r], outs[r], res_outs[r],
                           ps, pf, ms, flags[r], epochs[r], WORLD, T, H, EPS, GRID_X)
    graphs[r] = g
# replay all 8 graphs concurrently
for r in devs:
    torch.cuda.set_device(r)
    graphs[r].replay()
for r in devs:
    torch.cuda.synchronize(r)
t0 = time.perf_counter()
REP = 10
for _ in range(REP):
    for r in devs:
        torch.cuda.set_device(r)
        graphs[r].replay()
    for r in devs:
        torch.cuda.synchronize(r)
dt = (time.perf_counter() - t0) / REP / ROUNDS_IN_GRAPH * 1e6
print(f"AR+add+rmsnorm fused, T={T} H={H} world={WORLD}: {dt:.2f} us/op  (reference: flashinfer mnnvl fused in serving = 8.7 us)")
