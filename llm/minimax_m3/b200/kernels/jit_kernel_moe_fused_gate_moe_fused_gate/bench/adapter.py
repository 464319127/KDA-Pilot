"""Benchmark adapter for the MoE fused-gate router kernel.

Implements the API required by ``bench/benchmark.py`` (copied verbatim from the project
template). Baseline and candidate are invoked through the identical destination-passing
TVM-FFI ABI:

    moe_fused_gate(input, bias, output_out, indices_out, topk, scoring_func,
                   num_fused_shared_experts, renormalize, routed_scaling_factor,
                   apply_routed_scaling_factor_on_output)

neither call allocates output tensors (the caller pre-allocates ``output`` and ``indices``,
matching the upstream Python wrapper's allocation policy). The captured inputs are always
contiguous fp32 (all 296 captures report ``is_contiguous=True``), so inputs are created
contiguous here.
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


def make_case(workload: dict[str, Any], *, device: torch.device, seed: int) -> dict:
    shapes = workload["shapes"]
    scalars = workload["scalars"]
    N = int(shapes["num_tokens"])
    E = int(shapes["num_experts"])
    topk = int(scalars["topk"])

    # benchmark.py seeds torch before calling make_case; randn is deterministic.
    input_scores = torch.randn((N, E), dtype=torch.float32, device=device).contiguous()
    bias = torch.randn((E,), dtype=torch.float32, device=device).contiguous()

    inputs = {
        "input": input_scores,
        "bias": bias,
        "topk": topk,
        "scoring_func": int(scalars["scoring_func"]),  # 0 = sigmoid, 1 = sqrtsoftplus
        "num_fused_shared_experts": int(scalars["num_fused_shared_experts"]),
        "renormalize": bool(scalars["renormalize"]),
        "routed_scaling_factor": float(scalars["routed_scaling_factor"]),
        "apply_routed_scaling_factor_on_output": bool(
            scalars["apply_routed_scaling_factor_on_output"]
        ),
    }
    tolerance = {
        "atol": float(workload.get("atol", 1e-5)),
        "rtol": float(workload.get("rtol", 1e-5)),
    }
    # Plain dict (the template's _case_get handles dict access); avoids the dataclass +
    # string-annotation resolution issue when benchmark.py loads this adapter via importlib.
    return {
        "inputs": inputs,
        "baseline_outputs": _alloc_outputs(N, topk, device),
        "candidate_outputs": _alloc_outputs(N, topk, device),
        "tolerance": tolerance,
    }


def _invoke(mod, inputs: dict, outputs: tuple) -> None:
    output, indices = outputs
    mod.moe_fused_gate(
        inputs["input"],
        inputs["bias"],
        output,
        indices,
        inputs["topk"],
        inputs["scoring_func"],
        inputs["num_fused_shared_experts"],
        inputs["renormalize"],
        inputs["routed_scaling_factor"],
        inputs["apply_routed_scaling_factor_on_output"],
    )


# Resolve candidate availability ONCE at import — never inside the timed call.
# (`has_candidate()` does a filesystem stat; calling it per invocation adds an asymmetric
# ~µs syscall to the candidate side only, which silently biases the CPU-launch-bound decode
# timings under host/IO contention. Both call paths below now do only a cached module
# lookup + launch, so the A/B comparison is fair.)
_CANDIDATE_AVAILABLE = has_candidate()


def call_baseline(workload: dict[str, Any], inputs: dict, outputs: tuple) -> None:
    _invoke(baseline_module(), inputs, outputs)


def call_candidate(workload: dict[str, Any], inputs: dict, outputs: tuple) -> None:
    # Falls back to the baseline module if the native-CUDA candidate is absent, so the
    # harness is runnable end-to-end before the candidate is written.
    _invoke(candidate_module() if _CANDIDATE_AVAILABLE else baseline_module(), inputs, outputs)


def compare_outputs(
    workload: dict[str, Any],
    baseline_outputs: tuple,
    candidate_outputs: tuple,
    tolerance: dict,
) -> dict:
    """Top-k contract: exact-match on (ordered) selected indices; weights within tolerance.
    Indices are exact-match, not atol/rtol."""
    import math

    bv, bi = baseline_outputs
    cv, ci = candidate_outputs

    if bi.shape != ci.shape or bv.shape != cv.shape:
        return {"ok": False, "max_abs": math.inf, "max_rel": math.inf,
                "message": f"shape mismatch ids {tuple(bi.shape)}vs{tuple(ci.shape)} vals {tuple(bv.shape)}vs{tuple(cv.shape)}"}
    if ci.dtype != torch.int32 or cv.dtype != torch.float32:
        return {"ok": False, "max_abs": math.inf, "max_rel": math.inf,
                "message": f"dtype mismatch ids={ci.dtype} vals={cv.dtype}"}
    if torch.isnan(cv).any() or torch.isinf(cv).any():
        return {"ok": False, "max_abs": math.inf, "max_rel": math.inf,
                "message": "candidate weights contain NaN/Inf"}

    if not torch.equal(bi, ci):
        mism = int((bi != ci).sum().item())
        return {"ok": False, "max_abs": math.inf, "max_rel": math.inf,
                "message": f"ordered topk indices differ in {mism} slots (exact-match required)"}

    diff = (bv.float() - cv.float()).abs()
    denom = bv.float().abs().clamp_min(1e-12)
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    atol = float(tolerance.get("atol", 1e-5))
    rtol = float(tolerance.get("rtol", 1e-5))
    ok = bool(torch.all(diff <= (atol + rtol * bv.float().abs())).item())
    return {"ok": ok, "max_abs": max_abs, "max_rel": max_rel,
            "message": "" if ok else f"weights exceed tol atol={atol} rtol={rtol}"}
