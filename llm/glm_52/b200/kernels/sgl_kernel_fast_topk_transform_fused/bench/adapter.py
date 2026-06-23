"""Benchmark adapter for `fast_topk_transform_fused` (GLM-5.2 / B200).

Implements the adapter API consumed by bench/benchmark.py:
    make_case(workload, *, device, seed) -> Case
    call_baseline(workload, inputs, outputs) -> None
    call_candidate(workload, inputs, outputs) -> None
    compare_outputs(workload, baseline_outputs, candidate_outputs, tolerance) -> dict

The outputs are int32 page-table indices, so `compare_outputs` does EXACT integer
match (the harness default float atol/rtol comparator is wrong for index outputs).

PROVISIONAL (recovery-and-scaffold round): input synthesis covers the frozen
workload schema (random/ties scores, varied valid lengths, contiguous/strided
score, decode B==S and prefill B>S token layouts). It is finalized once the
remote build confirms the exact valid-range mapping and the returned-tensor count
(upstream main returns one (B,topk) tensor; the GLM-5.2 capture logged two — see
docs/baseline_source.md). call_baseline/call_candidate bind to the task-local ABI
that the build round produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

PAD_VALUE = -1  # baseline naive_topk_transform pads the tail [length, topk) with -1


# --------------------------------------------------------------------------- #
# Task-local ABI handle (built on the remote B200 in the build round).
# Baseline and candidate are exposed through the SAME signature / stream /
# destination-passing output policy, so timing is fair.
# --------------------------------------------------------------------------- #
_ABI = None


def _load_abi():
    global _ABI
    if _ABI is None:
        try:
            import topk_transform_abi as _abi  # built extension on the task path
        except Exception as exc:  # pragma: no cover - until the build round
            raise NotImplementedError(
                "task ABI not built yet: build baseline/ and solution/ on the "
                "remote B200 and expose fast_topk_transform_fused_baseline and "
                "fast_topk_transform_fused_candidate through `topk_transform_abi` "
                "(destination-passing, at::cuda::getCurrentCUDAStream()). "
                f"import error: {exc!r}"
            )
        _ABI = _abi
    return _ABI


@dataclass
class Case:
    inputs: dict[str, Any]
    baseline_outputs: list[torch.Tensor]
    candidate_outputs: list[torch.Tensor]
    tolerance: dict[str, float]


def _gen(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def _make_score(B, N, contiguous, dist, device, g) -> torch.Tensor:
    if dist == "ties":
        # Few distinct values -> many equal scores -> exercises tie-break ordering.
        score = torch.randint(0, 8, (B, N), device=device, generator=g).to(torch.float32)
    else:
        score = torch.randn(B, N, device=device, generator=g, dtype=torch.float32)
    if contiguous:
        return score.contiguous()
    # Reproduce a non-contiguous score: a column slice of a wider buffer has
    # row-stride N+pad != N, so is_contiguous() is False on dim 1.
    wide = torch.empty(B, N + 8, device=device, dtype=torch.float32)
    wide[:, :N] = score
    view = wide[:, :N]
    assert not view.is_contiguous() or B == 1
    return view


def _make_lengths(S, N, mode, device, g) -> torch.Tensor:
    if mode == "zero":
        vals = torch.zeros(S, device=device, dtype=torch.int32)
    elif mode == "one":
        vals = torch.ones(S, device=device, dtype=torch.int32)
    elif mode == "half":
        vals = torch.full((S,), max(1, N // 2), device=device, dtype=torch.int32)
    else:  # "full": every candidate valid
        vals = torch.full((S,), N, device=device, dtype=torch.int32)
    return vals


def _make_cu_seqlens(B, S, device) -> torch.Tensor:
    # Distribute B token-rows across S sequences as evenly as possible.
    if S == B:
        return torch.arange(0, S + 1, device=device, dtype=torch.int32)
    base = B // S
    rem = B % S
    counts = [base + (1 if i < rem else 0) for i in range(S)]
    cu = [0]
    for c in counts:
        cu.append(cu[-1] + c)
    return torch.tensor(cu, device=device, dtype=torch.int32)


def _make_page_table(S, M, device, g) -> torch.Tensor:
    # Valid logical->physical page-1 map; baseline and candidate share it, so the
    # transform result is deterministic regardless of the specific mapping.
    return torch.arange(S * M, device=device, dtype=torch.int32).reshape(S, M)


def make_case(workload: dict[str, Any], *, device: torch.device, seed: int) -> Case:
    sc = workload["scalars"]
    B, N, S, M = int(sc["B"]), int(sc["N"]), int(sc["S"]), int(sc["M"])
    topk = int(sc["topk"])
    contiguous = bool(workload["strides"]["score_contiguous"])
    dist = sc.get("score_dist", "random")
    lengths_mode = sc.get("lengths_mode", "full")
    g = _gen(device, seed)

    score = _make_score(B, N, contiguous, dist, device, g)
    lengths = _make_lengths(S, N, lengths_mode, device, g)
    page_table_size_1 = _make_page_table(S, M, device, g)
    cu_seqlens_q = _make_cu_seqlens(B, S, device)
    row_starts = None  # always None across the GLM-5.2 capture

    inputs = {
        "score": score,
        "lengths": lengths,
        "page_table_size_1": page_table_size_1,
        "cu_seqlens_q": cu_seqlens_q,
        "topk": topk,
        "row_starts": row_starts,
    }
    # Destination-passing output buffers (allocated once, excluded from timing).
    baseline_outputs = [torch.empty(B, topk, device=device, dtype=torch.int32)]
    candidate_outputs = [torch.empty(B, topk, device=device, dtype=torch.int32)]
    # Integer index outputs -> exact match.
    tolerance = {"atol": 0.0, "rtol": 0.0}
    return Case(inputs, baseline_outputs, candidate_outputs, tolerance)


def call_baseline(workload, inputs, outputs) -> None:
    abi = _load_abi()
    abi.fast_topk_transform_fused_baseline(
        inputs["score"], inputs["lengths"], inputs["page_table_size_1"],
        inputs["cu_seqlens_q"], inputs["topk"], inputs["row_starts"], outputs[0],
    )


def call_candidate(workload, inputs, outputs) -> None:
    abi = _load_abi()
    abi.fast_topk_transform_fused_candidate(
        inputs["score"], inputs["lengths"], inputs["page_table_size_1"],
        inputs["cu_seqlens_q"], inputs["topk"], inputs["row_starts"], outputs[0],
    )


def compare_outputs(workload, baseline_outputs, candidate_outputs, tolerance) -> dict:
    """Exact integer match on every output tensor (no atol/rtol on indices)."""
    if len(baseline_outputs) != len(candidate_outputs):
        return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                "message": f"output count mismatch {len(baseline_outputs)} vs {len(candidate_outputs)}"}
    for idx, (b, c) in enumerate(zip(baseline_outputs, candidate_outputs)):
        if b.shape != c.shape:
            return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                    "message": f"output {idx} shape {tuple(b.shape)} vs {tuple(c.shape)}"}
        if b.dtype != c.dtype:
            return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                    "message": f"output {idx} dtype {b.dtype} vs {c.dtype}"}
        if not torch.equal(b, c):
            mism = int((b != c).sum().item())
            return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                    "message": f"output {idx} exact-match failed: {mism} mismatched entries"}
    return {"ok": True, "max_abs": 0.0, "max_rel": 0.0, "message": ""}
