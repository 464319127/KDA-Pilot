"""Benchmark adapter for b200_tilert_head_proj_gemm.
Builds inputs and exposes the two ABI calls (baseline PyTorch / candidate CUDA)
for the standard bench template. Set TILERT_CANDIDATE_ALIAS_BASELINE=1 for A/A."""
from __future__ import annotations
import os, sys, torch
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TASK_DIR not in sys.path: sys.path.insert(0, TASK_DIR)
from baseline.head_proj import head_proj_baseline  # noqa: E402

_ALIAS = os.environ.get("TILERT_CANDIDATE_ALIAS_BASELINE", "0") == "1"
if not _ALIAS:
    try:
        from solution.binding import head_proj_candidate  # noqa: E402 (KDA agent writes this)
    except Exception:
        head_proj_candidate = None
else:
    head_proj_candidate = head_proj_baseline

def make_inputs(shapes, device):
    s, K, V = shapes["seq"], shapes["K"], shapes["V"]
    g = torch.Generator(device=device).manual_seed(0)
    h = torch.randn(s, K, device=device, dtype=torch.bfloat16, generator=g) / (K ** 0.5)
    W = torch.randn(V, K, device=device, dtype=torch.bfloat16, generator=g) / (K ** 0.5)
    return {"hidden": h, "W": W}

def call_baseline(inp):   return head_proj_baseline(inp["hidden"], inp["W"])
def call_candidate(inp):
    assert head_proj_candidate is not None, "solution/binding.py:head_proj_candidate missing (write the CUDA kernel)"
    return head_proj_candidate(inp["hidden"], inp["W"])
