#!/usr/bin/env python3
"""Profiling launcher for ncu: run ONE shape through baseline or candidate.
Usage: prof_one.py <baseline|candidate> <M> <K> <N>
Warms up (so JIT build + cuBLAS/CUTLASS init are excluded), then runs the target
once. Pair with: ncu --launch-skip 20 -c 1 -k regex:<kernel> ... python prof_one.py ...
"""
import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import adapter  # noqa: E402
import build_ext  # noqa: E402

side, M, K, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
wl = {"shapes": {"M": M, "K": K, "N": N}, "strides": {"b": "column_major"},
      "scalars": {"out_dtype": "bfloat16"}, "seed": 0}
inp = adapter.make_inputs(wl, device=torch.device("cuda"), seed=0)
out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
fn = build_ext.baseline if side == "baseline" else build_ext.candidate
for _ in range(25):
    fn(inp["a"], inp["b"], inp["scale_a"], inp["scale_b"], out)
torch.cuda.synchronize()
fn(inp["a"], inp["b"], inp["scale_a"], inp["scale_b"], out)  # profiled launch
torch.cuda.synchronize()
print("done", side, M, K, N)
