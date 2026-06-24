"""Benchmark/correctness adapter for topk_sigmoid (consumed by benchmark.py).

Exposes make_case / call_baseline / call_candidate / compare_outputs over the ABI-agnostic
surface in build_ext.py (baseline / candidate / route). The op is destination-passing:
topk_weights [N,topk] fp32 and topk_indices [N,topk] int32 are written in place; gating_output
[N,E] and correction_bias [E] (fp32) are read-only inputs. topk_weights/topk_indices are
write-only (never read by the kernel), so the harness may poison them before each call to catch
partial/stale writes — no ring buffer needed.

compare_outputs enforces the top-k contract: selected expert indices are EXACT-match (the
default element-wise float comparator would wrongly apply atol/rtol to int32 ids); gathered
weights use fp32 tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import build_ext

_DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass
class Case:
    inputs: tuple
    baseline_outputs: tuple
    candidate_outputs: tuple
    tolerance: dict


def _make_gating(n: int, e: int, dtype: torch.dtype, contiguous: bool,
                 device: torch.device, gen: torch.Generator) -> torch.Tensor:
    if contiguous:
        return torch.randn((n, e), dtype=dtype, device=device, generator=gen)
    # Non-contiguous [n, e]: materialize [e, n] and transpose (stride (1, n)).
    base = torch.randn((e, n), dtype=dtype, device=device, generator=gen)
    g = base.t()
    assert not g.is_contiguous() and g.shape == (n, e)
    return g


def make_case(workload: dict[str, Any], *, device: torch.device, seed: int) -> Case:
    n = int(workload["num_tokens"])
    e = int(workload["num_experts"])
    topk = int(workload["topk"])
    dtype = _DTYPE[workload.get("dtype", "float32")]
    contiguous = bool(workload.get("contiguous", True))
    renorm = 1 if bool(workload.get("renormalize", True)) else 0

    gen = torch.Generator(device=device).manual_seed(int(seed))
    gating = _make_gating(n, e, dtype, contiguous, device, gen)
    # correction_bias is always fp32 and present (captured contract). Synthetic, seeded;
    # randn magnitude keeps sigmoid+bias > -1 for far more than topk experts, so the recovered
    # baseline's genuine top-k selection holds (see docs/baseline_source.md).
    bias = torch.randn((e,), dtype=torch.float32, device=device, generator=gen)

    def fresh_outputs() -> tuple:
        w = torch.empty((n, topk), dtype=torch.float32, device=device)
        idx = torch.empty((n, topk), dtype=torch.int32, device=device)
        return (w, idx)

    return Case(
        inputs=(gating, renorm, bias),
        baseline_outputs=fresh_outputs(),
        candidate_outputs=fresh_outputs(),
        tolerance={"atol": float(workload.get("atol", 1e-5)), "rtol": float(workload.get("rtol", 1e-5))},
    )


def call_baseline(workload: dict[str, Any], inputs, outputs) -> None:
    gating, renorm, bias = inputs
    weights, indices = outputs
    build_ext.baseline(weights, indices, gating, renorm, bias)


def call_candidate(workload: dict[str, Any], inputs, outputs) -> None:
    gating, renorm, bias = inputs
    weights, indices = outputs
    build_ext.candidate(weights, indices, gating, renorm, bias)


def compare_outputs(workload, baseline_outputs, candidate_outputs, tolerance) -> dict[str, Any]:
    w_b, idx_b = baseline_outputs
    w_c, idx_c = candidate_outputs

    # Selected expert ids: EXACT match (integer/top-k contract).
    if idx_b.shape != idx_c.shape:
        return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                "message": f"index shape mismatch {tuple(idx_b.shape)} vs {tuple(idx_c.shape)}"}
    if not torch.equal(idx_b.to(torch.int64), idx_c.to(torch.int64)):
        n_diff = int((idx_b.to(torch.int64) != idx_c.to(torch.int64)).sum().item())
        return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                "message": f"top-k index mismatch: {n_diff} differing entries (exact match required)"}

    # Gathered weights: fp32 tolerance, with NaN/Inf guard.
    if not (torch.isfinite(w_c).all().item()):
        return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                "message": "candidate weights contain NaN/Inf"}
    atol = float(tolerance.get("atol", 1e-5))
    rtol = float(tolerance.get("rtol", 1e-5))
    diff = (w_b.float() - w_c.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = w_b.float().abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    ok = torch.allclose(w_c.float(), w_b.float(), atol=atol, rtol=rtol)
    return {"ok": bool(ok), "max_abs": max_abs, "max_rel": max_rel,
            "message": "" if ok else f"weight mismatch max_abs={max_abs:.3e} max_rel={max_rel:.3e} (atol={atol},rtol={rtol})"}
