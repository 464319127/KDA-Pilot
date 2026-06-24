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


def _candidate_in_domain(workload: dict[str, Any]) -> bool:
    """Exactly the candidate kernel's `in_domain` gate — the ONLY config for which the candidate
    runs its cold-safe warp-per-token kernel (rather than routing to the copied baseline fallback).
    Must mirror the predicate in solution/csrc/moe/moe_fused_gate_candidate.cuh."""
    sh = workload.get("shapes", {})
    sc = workload.get("scalars", {})
    return (int(sh.get("num_experts", 0)) == 128
            and int(sc.get("topk", 0)) == 5
            and int(sc.get("scoring_func", 0)) == 0
            and int(sc.get("num_fused_shared_experts", 0)) == 1
            and bool(sc.get("renormalize", False))
            and float(sc.get("routed_scaling_factor", 0.0)) == 2.0
            and bool(sc.get("apply_routed_scaling_factor_on_output", False)))


def _baseline_decode_is_ub(workload: dict[str, Any]) -> bool:
    """The recovered baseline's small-token (decode) kernel reads uninitialized shared memory
    whenever it is dispatched to a warp-count template (<8>/<12>/<16>) with fewer active warps —
    i.e. when warps_per_token = min(ceil(E/32), 16) is not exactly 8, 12, or 16. That covers E=128
    (warps_per_token=4) and other ranges. Such a launch can raise CUDA illegal-memory-access on a
    cold context (docs/baseline_source.md). Decode = small-token dispatch (num_rows <= 512)."""
    sh = workload.get("shapes", {})
    if int(sh.get("num_tokens", 1)) > 512:
        return False
    warps_per_token = min((int(sh.get("num_experts", 0)) + 31) // 32, 16)
    return warps_per_token not in (8, 12, 16)


_warned_ub_decode = False


def call_baseline(workload: dict[str, Any], inputs: dict, outputs: tuple) -> None:
    # The baseline decode path can be UB (uninitialized shared-memory read) and unsafe to launch.
    # Only the candidate's EXACT in_domain config has a cold-safe replacement (its warp-per-token
    # kernel); substitute the candidate there so the harness does not fault. The resulting decode
    # A/B speedup is degenerate (candidate-vs-candidate) and the row is `production:false` (excluded
    # from the headline); authoritative decode evidence is candidate-only
    # (bench/bench_decode_candidate.py, docs/results.md).
    if _baseline_decode_is_ub(workload):
        if _candidate_in_domain(workload) and _CANDIDATE_AVAILABLE:
            global _warned_ub_decode
            if not _warned_ub_decode:
                import sys
                print("[adapter] baseline decode path is UB; substituting the cold-safe candidate "
                      "for the captured in_domain config. Decode A/B speedups are degenerate — use "
                      "bench_decode_candidate.py for decode latency (see docs/results.md).",
                      file=sys.stderr)
                _warned_ub_decode = True
            _invoke(candidate_module(), inputs, outputs)
            return
        # Off-domain UB decode: the baseline is UB AND the candidate would route to its verbatim
        # baseline fallback (the same UB). Neither side is cold-safe, so this shape is
        # unbenchmarkable. Refuse cleanly BEFORE any CUDA launch (no context poison) rather than
        # crash. (No frozen workloads.json row hits this — all captured decode rows are in_domain.)
        sh = workload.get("shapes", {})
        raise RuntimeError(
            f"off-domain decode shape (num_experts={sh.get('num_experts')}, "
            f"num_tokens={sh.get('num_tokens')}, scalars outside the candidate-safe domain) is "
            f"unbenchmarkable: the baseline small-token path is UB and the candidate falls back to "
            f"it. Use the captured in_domain config, a prefill (M>512) shape, or an E with "
            f"warps_per_token in {{8,12,16}}.")
    _invoke(baseline_module(), inputs, outputs)


def call_candidate(workload: dict[str, Any], inputs: dict, outputs: tuple) -> None:
    # Falls back to the baseline module if the native-CUDA candidate is absent, so the
    # harness is runnable end-to-end before the candidate is written.
    _invoke(candidate_module() if _CANDIDATE_AVAILABLE else baseline_module(), inputs, outputs)


# NOTE: there is deliberately NO import-time context priming. The JIT modules build lazily on the
# first call_baseline / call_candidate — which happens inside benchmark.py's workload loop, AFTER it
# runs torch.cuda.set_device(args.device) — so they compile for and run on the SELECTED device, not
# the default GPU at import time (important for non-default --device / heterogeneous hosts). Priming
# is unnecessary for safety: the baseline decode path is never launched (call_baseline substitutes
# the cold-safe candidate for the in_domain config and errors for off-domain UB decode), the baseline
# large-token path and the candidate are cold-safe, and benchmark.py warms each workload
# (warmup_runs) before the timed region.


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
