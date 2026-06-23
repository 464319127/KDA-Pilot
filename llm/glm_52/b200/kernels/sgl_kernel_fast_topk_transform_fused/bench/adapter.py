"""Benchmark adapter for `fast_topk_transform_fused` (GLM-5.2 / B200).

Implements the adapter API consumed by bench/benchmark.py:
    make_case(workload, *, device, seed) -> Case
    call_baseline(workload, inputs, outputs) -> None
    call_candidate(workload, inputs, outputs) -> None
    compare_outputs(workload, baseline_outputs, candidate_outputs, tolerance) -> dict

Faithful to the recovered C++ contract (baseline/.../elementwise/topk.cu):
  - `lengths` is shape (B,) per token row (C++ asserts lengths.size(0)==score.size(0)),
    capped to min(N, M) so the baseline naive path never reads past `src_page_table`/score.
  - `page_table_size_1` (src_page_table) is shape (S, M); S = cu_seqlens_q.size(0)-1.
  - is_decode = (row_starts is None and S == B); else prefill (cu_seqlens maps token->seq).
  - `score` has stride(1)==1; "non-contiguous" captures are row-strided (stride(0) > N).
  - ONE output (dst_page_table (B, topk) int32). The capture's two identical records are
    logger duplication (see docs/baseline_source.md); final confirmation = remote probe.

The outputs are int32 indices, so `compare_outputs` is EXACT integer match (the harness
default float atol/rtol comparator is wrong for index outputs). call_baseline/call_candidate
bind to the task-local `topk_transform_abi` module built on the remote B200 (see solution/).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

PAD_VALUE = -1  # baseline naive_topk_transform pads tail [length, topk) with -1
ROW_PAD = 8     # surrogate row padding for non-contiguous score (matches gen_workloads.py)

_ABI = None


def _load_abi():
    """Lazy handle to the task-local built ABI (baseline + candidate through the same
    destination-passing signature). Built on the remote B200 (see solution/build.py)."""
    global _ABI
    if _ABI is None:
        import sys
        from pathlib import Path
        sol = Path(__file__).resolve().parent.parent / "solution"
        if str(sol) not in sys.path:
            sys.path.insert(0, str(sol))  # find the .so built by solution/build.py
        try:
            import topk_transform_abi as _abi
        except Exception as exc:  # pragma: no cover - until the remote build
            raise NotImplementedError(
                "task ABI not built yet: run solution/build.py on the remote B200 to "
                "compile baseline + candidate + binding into `topk_transform_abi` "
                "(fast_topk_transform_fused_baseline / _candidate, destination-passing, "
                f"at::cuda::getCurrentCUDAStream()). import error: {exc!r}"
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


def make_score(B, N, contiguous, dist, device, g) -> torch.Tensor:
    """score (B, N) float32 with stride(1)==1. Non-contiguous => row-strided
    (stride(0) = N + ROW_PAD > N), matching the captured is_contiguous=False rows."""
    if dist == "ties":
        vals = torch.randint(0, 8, (B, N), device=device, generator=g).to(torch.float32)
    else:
        vals = torch.randn(B, N, device=device, generator=g, dtype=torch.float32)
    if contiguous:
        out = vals.contiguous()
    else:
        wide = torch.empty(B, N + ROW_PAD, device=device, dtype=torch.float32)
        wide[:, :N] = vals
        out = wide[:, :N]  # shape (B,N), stride (N+ROW_PAD, 1): stride(1)==1, stride(0)>N
    assert out.stride(1) == 1, "score must be contiguous on dim 1 (C++ requires stride(1)==1)"
    return out


def make_lengths(B, N, M, mode, device) -> torch.Tensor:
    """lengths (B,) int32, capped to min(N, M) so naive path never over-reads."""
    cap = min(N, M)
    if mode == "zero":
        val = 0
    elif mode == "one":
        val = min(1, cap)
    elif mode == "half":
        val = max(0, cap // 2)
    else:  # "full"
        val = cap
    return torch.full((B,), val, device=device, dtype=torch.int32)


def make_cu_seqlens(B, S, device) -> torch.Tensor:
    """cu_seqlens_q (S+1,) int32. Decode: S==B -> arange. Prefill: distribute B over S."""
    if S == B:
        return torch.arange(0, S + 1, device=device, dtype=torch.int32)
    base, rem = divmod(B, S)
    cu = [0]
    for i in range(S):
        cu.append(cu[-1] + base + (1 if i < rem else 0))
    return torch.tensor(cu, device=device, dtype=torch.int32)


def make_page_table(S, M, device) -> torch.Tensor:
    """src_page_table (S, M) int32 logical->physical map (contiguous on dim 1)."""
    return torch.arange(S * M, device=device, dtype=torch.int32).reshape(S, M)


def make_case(workload: dict[str, Any], *, device: torch.device, seed: int) -> Case:
    sc = workload["scalars"]
    B, N, S, M = int(sc["B"]), int(sc["N"]), int(sc["S"]), int(sc["M"])
    topk = int(sc["topk"])
    contiguous = bool(workload["strides"]["score_contiguous"])
    dist = sc.get("score_dist", "random")
    lengths_mode = sc.get("lengths_mode", "full")
    g = _gen(device, seed)

    inputs = {
        "score": make_score(B, N, contiguous, dist, device, g),
        "lengths": make_lengths(B, N, M, lengths_mode, device),
        "page_table_size_1": make_page_table(S, M, device),
        "cu_seqlens_q": make_cu_seqlens(B, S, device),
        "topk": topk,
        "row_starts": None,  # always None across the GLM-5.2 capture
    }
    baseline_outputs = [torch.empty(B, topk, device=device, dtype=torch.int32)]
    candidate_outputs = [torch.empty(B, topk, device=device, dtype=torch.int32)]
    return Case(inputs, baseline_outputs, candidate_outputs, {"atol": 0.0, "rtol": 0.0})


def call_baseline(workload, inputs, outputs) -> None:
    _load_abi().fast_topk_transform_fused_baseline(
        inputs["score"], inputs["lengths"], inputs["page_table_size_1"],
        inputs["cu_seqlens_q"], inputs["topk"], inputs["row_starts"], outputs[0])


def call_candidate(workload, inputs, outputs) -> None:
    _load_abi().fast_topk_transform_fused_candidate(
        inputs["score"], inputs["lengths"], inputs["page_table_size_1"],
        inputs["cu_seqlens_q"], inputs["topk"], inputs["row_starts"], outputs[0])


def compare_outputs(workload, baseline_outputs, candidate_outputs, tolerance) -> dict:
    """Exact integer match on every output (no atol/rtol on indices), with shape/dtype/
    contiguity checks. Int32 outputs cannot hold NaN/Inf; a finite check is applied only
    to any float output for completeness."""
    if len(baseline_outputs) != len(candidate_outputs):
        return _fail(f"output count {len(baseline_outputs)} vs {len(candidate_outputs)}")
    for idx, (b, c) in enumerate(zip(baseline_outputs, candidate_outputs)):
        if b.shape != c.shape:
            return _fail(f"output {idx} shape {tuple(b.shape)} vs {tuple(c.shape)}")
        if b.dtype != c.dtype:
            return _fail(f"output {idx} dtype {b.dtype} vs {c.dtype}")
        if b.stride() != c.stride():
            return _fail(f"output {idx} stride {b.stride()} vs {c.stride()}")
        if c.is_floating_point() and not torch.isfinite(c).all():
            return _fail(f"output {idx} has NaN/Inf")
        if not torch.equal(b, c):
            mism = int((b != c).sum().item())
            return _fail(f"output {idx} exact-match failed: {mism} mismatched entries")
    return {"ok": True, "max_abs": 0.0, "max_rel": 0.0, "message": ""}


def _fail(msg: str) -> dict:
    return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"), "message": msg}
