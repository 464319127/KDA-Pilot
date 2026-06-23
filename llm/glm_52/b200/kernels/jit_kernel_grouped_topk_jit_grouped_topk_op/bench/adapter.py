"""Benchmark adapter for the grouped-top-k router kernel.

Implements the API required by ``bench/benchmark.py`` (copied verbatim from the
project template). Baseline and candidate are invoked through the identical
destination-passing TVM-FFI ABI (`grouped_topk(scores, bias, topk_values_out,
topk_indices_out, num_expert_group, topk_group, topk, renormalize,
scaling_factor)`); neither call allocates output tensors.

The op's captured inputs are always contiguous fp32 (the upstream Python wrapper
calls ``.contiguous()`` before the op, and all 17404 captures report
``is_contiguous=True``), so inputs are created contiguous here.
"""

from __future__ import annotations

from typing import Any

import torch

from _jit_build import baseline_module, candidate_module, has_candidate


def _alloc_outputs(num_tokens: int, topk: int, device: torch.device) -> tuple:
    return (
        torch.empty((num_tokens, topk), dtype=torch.float32, device=device),
        torch.empty((num_tokens, topk), dtype=torch.int32, device=device),
    )


def make_case(workload: dict[str, Any], *, device: torch.device, seed: int) -> Case:
    shapes = workload["shapes"]
    scalars = workload["scalars"]
    N = int(shapes["num_tokens"])
    E = int(shapes["num_experts"])
    topk = int(scalars["topk"])

    # benchmark.py seeds torch before calling make_case; randn is deterministic.
    scores = torch.randn((N, E), dtype=torch.float32, device=device).contiguous()
    bias = torch.randn((E,), dtype=torch.float32, device=device).contiguous()

    inputs = {
        "scores": scores,
        "bias": bias,
        "num_expert_group": int(scalars["num_expert_group"]),
        "topk_group": int(scalars["topk_group"]),
        "topk": topk,
        "renormalize": bool(scalars["renormalize"]),
        "scaling_factor": float(scalars["scaling_factor"]),
    }
    tolerance = {
        "atol": float(workload.get("atol", 1e-5)),
        "rtol": float(workload.get("rtol", 1e-5)),
    }
    # Plain dict (the template's _case_get handles dict access); avoids the
    # dataclass + string-annotation resolution issue when benchmark.py loads
    # this adapter via importlib (module is not registered in sys.modules).
    return {
        "inputs": inputs,
        "baseline_outputs": _alloc_outputs(N, topk, device),
        "candidate_outputs": _alloc_outputs(N, topk, device),
        "tolerance": tolerance,
    }


def _invoke(mod, inputs: dict, outputs: tuple) -> None:
    topk_values, topk_indices = outputs
    mod.grouped_topk(
        inputs["scores"],
        inputs["bias"],
        topk_values,
        topk_indices,
        inputs["num_expert_group"],
        inputs["topk_group"],
        inputs["topk"],
        inputs["renormalize"],
        inputs["scaling_factor"],
    )


# Resolve candidate availability ONCE at import — never inside the timed call.
# (`has_candidate()` does a filesystem stat; calling it per invocation adds an
# asymmetric ~µs syscall to the candidate side only, which silently biases the
# CPU-launch-bound decode timings under host/IO contention. Both call paths below
# now do only a cached module lookup + launch, so the A/B comparison is fair.)
_CANDIDATE_AVAILABLE = has_candidate()


def call_baseline(workload: dict[str, Any], inputs: dict, outputs: tuple) -> None:
    _invoke(baseline_module(), inputs, outputs)


def call_candidate(workload: dict[str, Any], inputs: dict, outputs: tuple) -> None:
    # Falls back to the baseline module if the native-CUDA candidate is absent, so
    # the harness is runnable end-to-end before the candidate is written.
    _invoke(candidate_module() if _CANDIDATE_AVAILABLE else baseline_module(), inputs, outputs)


def compare_outputs(
    workload: dict[str, Any],
    baseline_outputs: tuple,
    candidate_outputs: tuple,
    tolerance: dict,
) -> dict:
    """Top-k contract: exact-match on (ordered) selected indices; weights within
    tolerance. Indices are exact-match, not atol/rtol."""
    bv, bi = baseline_outputs
    cv, ci = candidate_outputs
    import math

    if bi.shape != ci.shape or bv.shape != cv.shape:
        return {"ok": False, "max_abs": math.inf, "max_rel": math.inf,
                "message": f"shape mismatch ids {tuple(bi.shape)}vs{tuple(ci.shape)} vals {tuple(bv.shape)}vs{tuple(cv.shape)}"}
    if ci.dtype != torch.int32 or cv.dtype != torch.float32:
        return {"ok": False, "max_abs": math.inf, "max_rel": math.inf,
                "message": f"dtype mismatch ids={ci.dtype} vals={cv.dtype}"}
    if torch.isnan(cv).any() or torch.isinf(cv).any():
        return {"ok": False, "max_abs": math.inf, "max_rel": math.inf,
                "message": "candidate weights contain NaN/Inf"}

    idx_equal = torch.equal(bi, ci)
    if not idx_equal:
        mism = int((bi != ci).sum().item())
        return {"ok": False, "max_abs": math.inf, "max_rel": math.inf,
                "message": f"ordered topk_indices differ in {mism} slots (exact-match required)"}

    diff = (bv.float() - cv.float()).abs()
    denom = bv.float().abs().clamp_min(1e-12)
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    atol = float(tolerance.get("atol", 1e-5))
    rtol = float(tolerance.get("rtol", 1e-5))
    ok = bool(torch.all(diff <= (atol + rtol * bv.float().abs())).item())
    return {"ok": ok, "max_abs": max_abs, "max_rel": max_rel,
            "message": "" if ok else f"topk_values exceed tol atol={atol} rtol={rtol}"}
