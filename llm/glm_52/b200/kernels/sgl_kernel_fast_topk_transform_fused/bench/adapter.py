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
  - ONE output (dst_page_table (B, topk) int32), confirmed on B200 (Round 2 probe). The capture's
    two result records per variant are two sampled calls' single returns (multi-call aggregation,
    not two outputs of one call; see docs/baseline_source.md).

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
        import importlib.util
        from pathlib import Path
        build_py = Path(__file__).resolve().parent.parent / "solution" / "build.py"
        try:
            spec = importlib.util.spec_from_file_location("topk_abi_build", build_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _ABI = mod.load_abi()  # builds (cached) + loads the TVM-FFI module
        except Exception as exc:  # pragma: no cover - until the remote build
            raise NotImplementedError(
                "task ABI not built: run `python solution/build.py` on the remote B200 "
                "(builds the TVM-FFI topk_transform_abi via tvm_ffi.cpp.load_inline). "
                f"error: {exc!r}"
            )
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


def make_page_table(S, M, device, g, mode="arange") -> torch.Tensor:
    """src_page_table (S, M) int32 logical->physical map (contiguous on dim 1). Each row holds
    distinct page ids (invertible). mode='permuted' makes it non-linear so the transform is
    actually validated (not just an arange that hides a base+pos shortcut)."""
    base = torch.arange(S * M, device=device, dtype=torch.int32).reshape(S, M)
    if mode == "permuted":
        out = torch.empty_like(base)
        for s in range(S):
            out[s] = base[s][torch.randperm(M, generator=g, device=device)]
        return out
    return base


def make_row_starts(B, N, L, kind, device):
    """Captured ragged-prefill row_starts: int32 (B,) with row_start[b] + length <= N so the
    baseline's score window score[b, row_start:row_start+length] stays in bounds. None otherwise."""
    if kind != "tensor":
        return None
    maxstart = max(0, N - L)
    if maxstart == 0:
        return torch.zeros(B, device=device, dtype=torch.int32)
    rs = (torch.arange(B, device=device, dtype=torch.int64) * 7) % (maxstart + 1)
    return rs.to(torch.int32)


def make_case(workload: dict[str, Any], *, device: torch.device, seed: int) -> Case:
    sc = workload["scalars"]
    B, N, S, M = int(sc["B"]), int(sc["N"]), int(sc["S"]), int(sc["M"])
    topk = int(sc["topk"])
    contiguous = bool(workload["strides"]["score_contiguous"])
    dist = sc.get("score_dist", "random")
    lengths_mode = sc.get("lengths_mode", "full")
    page_table_mode = sc.get("page_table_mode", "arange")
    row_starts_kind = sc.get("row_starts_kind", "none")
    g = _gen(device, seed)

    lengths = make_lengths(B, N, M, lengths_mode, device)
    valid_len = int(lengths.max().item()) if lengths.numel() else 0
    inputs = {
        "score": make_score(B, N, contiguous, dist, device, g),
        "lengths": lengths,
        "page_table_size_1": make_page_table(S, M, device, g, page_table_mode),
        "cu_seqlens_q": make_cu_seqlens(B, S, device),
        "topk": topk,
        # 243 captured variants have row_starts=None; 4 large-prefill variants pass a (B,) tensor.
        "row_starts": make_row_starts(B, N, valid_len, row_starts_kind, device),
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
    contiguity checks. This is the comparator for the DETERMINISTIC (naive, length<=topk)
    regime; the non-deterministic radix path is validated by correctness.validate_topk
    (valid-top-k), not by exact compare. Int32 outputs cannot hold NaN/Inf; a finite check
    is applied only to any float output for completeness."""
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
