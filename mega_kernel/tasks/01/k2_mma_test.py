"""K2v4 mma kernel: correctness + cold-L2 benchmark vs DeepGEMM-fp8 and cuBLAS-bf16."""
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load

torch.cuda.set_device(int(os.environ.get("K2_DEV", "7")))

ext = load(
    name="k2_mma",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "k2_mma_kernel.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_103a,code=sm_103a",
        "--use_fast_math",
    ],
    verbose=False,
)
k2_mma = ext


def k2_run(A, W, S, split_k=None, part=None, out=None):
    M, K = A.shape
    N = W.shape[0]
    KB = (K + 127) // 128
    n_tiles = (N + 63) // 64
    if split_k is None:
        split_k = min(KB, max(1, 192 // n_tiles))
    kb_per_split = (KB + split_k - 1) // split_k
    split_k = (KB + kb_per_split - 1) // kb_per_split
    if out is None:
        out = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    if split_k > 1:
        if part is None:
            part = torch.empty(split_k, 16, N, dtype=torch.float32, device=A.device)
        k2_mma.pass1(
            A, W, S, part, out, M, N, K, KB, kb_per_split, S.stride(0),
            part.stride(0), part.stride(1), split_k,
        )
        k2_mma.pass2(part, out, M, N, split_k, part.stride(0), part.stride(1))
    else:
        dummy = torch.empty(1, dtype=torch.float32, device=A.device)
        k2_mma.pass1(
            A, W, S, dummy, out, M, N, K, KB, kb_per_split, S.stride(0),
            0, 0, 1,
        )
    return out


def make_case(M, N, K, device="cuda"):
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = (torch.randn(N, K, dtype=torch.float32, device=device) * 0.05).to(torch.float8_e4m3fn)
    S = torch.rand((N + 127) // 128, (K + 127) // 128, dtype=torch.float32, device=device) * 0.5 + 0.5
    return A, W, S


def ref(A, W, S, K, N):
    Wd = W.to(torch.float32)
    s_exp = S.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
    return A.to(torch.float32) @ (Wd * s_exp).T


shapes = [(6, 2624, 6144), (6, 2048, 2048), (6, 6144, 2048), (6, 512, 6144),
          (6, 6144, 256), (6, 3072, 6144), (6, 6144, 1536), (1, 6144, 2048)]

print("== correctness ==")
for M, N, K in shapes:
    A, W, S = make_case(M, N, K)
    out = k2_run(A, W, S)
    r = ref(A, W, S, K, N)
    err = (out.float() - r).abs().max().item()
    rel = err / r.abs().max().item()
    print(f"M{M} N{N} K{K}: max_abs {err:.4f} rel {rel:.2e} {'OK' if rel < 2e-2 else 'FAIL'}")

print("== cold-L2 benchmark (48 weight copies, graph replay) ==")
NC = 48
for M, N, K in shapes:
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    Ws = [(torch.randn(N, K, dtype=torch.float32, device="cuda") * 0.05).to(torch.float8_e4m3fn) for _ in range(NC)]
    S = torch.rand((N + 127) // 128, (K + 127) // 128, dtype=torch.float32, device="cuda") * 0.5 + 0.5
    # preallocate stable buffers per call for graph
    KB = (K + 127) // 128
    n_tiles = (N + 63) // 64
    split_k = min(KB, max(1, 192 // n_tiles))
    kb_per_split = (KB + split_k - 1) // split_k
    split_k = (KB + kb_per_split - 1) // kb_per_split
    part = torch.empty(max(split_k, 1), 16, N, dtype=torch.float32, device="cuda")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    for w in Ws[:4]:
        k2_run(A, w, S, split_k=split_k, part=part, out=out)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for w in Ws:
            k2_run(A, w, S, split_k=split_k, part=part, out=out)
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    R = 10
    for _ in range(R):
        g.replay()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / R / NC * 1e6
    mb = N * K / 1e6
    print(f"M{M} N{N} K{K}: {us:6.2f} us/call ({mb:.1f}MB fp8, {mb/us/1e6:.2f} TB/s) split_k={split_k}")

    # bf16 cublas cold reference
    Wb = [torch.randn(N, K, dtype=torch.bfloat16, device="cuda") for _ in range(NC)]
    gb = torch.cuda.CUDAGraph()
    for w in Wb[:2]:
        torch.nn.functional.linear(A, w)
    torch.cuda.synchronize()
    with torch.cuda.graph(gb):
        for w in Wb:
            torch.nn.functional.linear(A, w)
    for _ in range(2):
        gb.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(R):
        gb.replay()
    torch.cuda.synchronize()
    us_b = (time.perf_counter() - t0) / R / NC * 1e6
    print(f"          cublas bf16 cold: {us_b:6.2f} us/call ({2*mb/us_b/1e6:.2f} TB/s)")
    del Ws, Wb
    torch.cuda.empty_cache()
