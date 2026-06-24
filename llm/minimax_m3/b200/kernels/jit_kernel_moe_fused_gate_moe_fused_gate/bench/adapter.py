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


def build_inputs(generator: str, N: int, E: int, device: torch.device, gen=None):
    """Construct (input, bias) fp32 contiguous tensors for a workload's `generator` name.

    `randn` is the production default (benchmark.py seeds torch globally first, so gen=None is
    deterministic). The semantic edge generators construct the exact AC-3 regression inputs from
    `bench/workloads.json` so that file is the authoritative frozen workload set. Logits may include
    +Inf/-Inf (the `pos_inf`/`neg_inf` edge rows); their sigmoid is finite (1/0), so candidate and
    oracle agree. NaN inputs are out of contract (never produced by any generator here)."""
    def _randn(shape):
        return torch.randn(shape, dtype=torch.float32, device=device, generator=gen)
    def _full(shape, v):
        return torch.full(shape, v, dtype=torch.float32, device=device)

    if generator == "randn":
        return _randn((N, E)).contiguous(), _randn((E,)).contiguous()
    if generator == "all_equal":          # equal logits -> selection decided by bias (+ tie-break)
        return _full((N, E), 0.3).contiguous(), _randn((E,)).contiguous()
    if generator == "saturate_neg":       # sigmoid ~ 0 -> routed_sum ~ 0 (norm fallback path)
        return _full((N, E), -60.0).contiguous(), torch.zeros((E,), dtype=torch.float32, device=device).contiguous()
    if generator == "saturate_pos":       # sigmoid ~ 1
        return _full((N, E), 60.0).contiguous(), torch.zeros((E,), dtype=torch.float32, device=device).contiguous()
    if generator == "pos_inf":            # sigmoid(+inf)=1 -> ties resolved by bias + smaller index
        return _full((N, E), float("inf")).contiguous(), _randn((E,)).contiguous()
    if generator == "neg_inf":            # sigmoid(-inf)=0 -> routed_sum=0; selection by bias
        return _full((N, E), float("-inf")).contiguous(), _randn((E,)).contiguous()
    if generator == "tie_small_index":    # exact tie on biased score -> smaller expert index wins
        inp = torch.zeros((N, E), dtype=torch.float32, device=device)
        bias = _full((E,), -1.0); bias[3] = 1.0; bias[100] = 1.0
        return inp.contiguous(), bias.contiguous()
    if generator == "subnormal_sum":      # sigmoid ~ 1e-37 -> tiny routed_sum (fp32 op-order stress)
        inp = _full((N, E), -85.0)
        bias = torch.zeros((E,), dtype=torch.float32, device=device); bias[7] = 1e-3; bias[40] = 5e-4
        return inp.contiguous(), bias.contiguous()
    raise ValueError(f"unknown workload generator: {generator!r}")


def make_case(workload: dict[str, Any], *, device: torch.device, seed: int) -> dict:
    shapes = workload["shapes"]
    scalars = workload["scalars"]
    N = int(shapes["num_tokens"])
    E = int(shapes["num_experts"])
    topk = int(scalars["topk"])

    # benchmark.py seeds torch before calling make_case; randn is deterministic. Edge rows in
    # workloads.json carry a `generator` name that selects the exact semantic input construction.
    input_scores, bias = build_inputs(workload.get("generator", "randn"), N, E, device)

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


def _prime_context() -> None:
    """Prime the CUDA context once before any timed region.

    The recovered baseline's decode (small-token) kernel reads uninitialized shared memory
    for num_experts=128 and faults on a COLD context (see docs/baseline_source.md). A safe
    large-path launch first primes the shared-memory region so the first *timed* decode call
    is warm — mirroring warmed serving. This runs at import (outside benchmark.py's timed
    region) and is symmetric: it builds + warms both modules the harness will call.
    """
    # Only the LARGE-path baseline launch is cold-safe. The baseline's small-token (decode)
    # kernel has an uninitialized-shared-memory bug for num_experts=128 that can fault
    # nondeterministically and poison the CUDA context — so we must NOT launch the baseline
    # decode path here. (The candidate is cold-safe and needs no priming.) See
    # docs/baseline_source.md. Decode-vs-baseline timing is therefore not obtainable; run
    # baseline benchmarks on prefill rows (--only ...prefill ids) and report candidate decode
    # as absolute timing with the baseline decode bug noted in docs/results.md.
    try:
        dev = torch.device("cuda")
        x = torch.randn((1024, 128), dtype=torch.float32, device=dev)
        b = torch.randn((128,), dtype=torch.float32, device=dev)
        o = torch.empty((1024, 5), dtype=torch.float32, device=dev)
        i = torch.empty((1024, 5), dtype=torch.int32, device=dev)
        baseline_module().moe_fused_gate(x, b, o, i, 5, 0, 1, True, 2.0, True)  # large path only
        if _CANDIDATE_AVAILABLE:
            candidate_module().moe_fused_gate(x, b, o, i, 5, 0, 1, True, 2.0, True)
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 — priming is best-effort; never block the harness import
        pass


if torch.cuda.is_available():
    _prime_context()


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
