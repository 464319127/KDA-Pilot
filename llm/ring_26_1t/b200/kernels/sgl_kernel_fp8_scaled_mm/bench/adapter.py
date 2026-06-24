"""Adapter for standalone_llm_benchmark_template.py: builds one fp8_scaled_mm
case (synthetic inputs + pre-allocated, poisoned outputs) and launches the
baseline / candidate through the shared destination-passing ABI (build_ext).

No tensor values are captured in docs/evidence.json, so inputs are synthetic
with a fixed per-workload seed:
  A      = [M,K] fp8_e4m3, row-major (contiguous)         <- randn -> fp8
  B      = [K,N] fp8_e4m3, COLUMN-MAJOR (stride (1,K))    <- randn[N,K].t() -> fp8
  scale_a= [M,1] fp32 positive  (per-token)
  scale_b= [N,1] fp32 positive  (per-channel)
  out    = [M,N] bf16 (or fp16 for the fp16-out edge row), row-major, poisoned

The timed region (in the template) excludes input generation and allocation;
both sides receive the SAME pre-allocated output object class and write in place.
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_ext  # noqa: E402

_DT = {"bfloat16": torch.bfloat16, "float16": torch.float16}
_POISON = float("nan")


def _make_fp8(shape, device, generator):
    # randn in fp8_e4m3 range (|x|<=448), then quantize.
    x = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    return x.to(torch.float8_e4m3fn)


def make_inputs(workload: dict, *, device: torch.device, seed: int) -> dict:
    s = workload["shapes"]
    M, K, N = s["M"], s["K"], s["N"]
    g = torch.Generator(device=device).manual_seed(int(seed) ^ (M * 1_000_003 + K * 1009 + N))
    a = _make_fp8((M, K), device, g)  # row-major [M,K]
    b_contiguous = workload.get("strides", {}).get("b") == "row_major"
    if b_contiguous:
        b = _make_fp8((K, N), device, g)  # contiguous [K,N] (edge: contract expects column-major)
    else:
        w = _make_fp8((N, K), device, g)  # [N,K] contiguous
        b = w.t()                         # [K,N] view, stride (1,K) == column-major
    scale_a = (torch.rand((M, 1), device=device, dtype=torch.float32, generator=g) * 0.02 + 1e-3)
    scale_b = (torch.rand((N, 1), device=device, dtype=torch.float32, generator=g) * 0.02 + 1e-3)
    return {"a": a, "b": b, "scale_a": scale_a, "scale_b": scale_b, "M": M, "K": K, "N": N}


def _alloc_out(workload: dict, device: torch.device) -> torch.Tensor:
    s = workload["shapes"]
    out_dt = _DT[workload.get("scalars", {}).get("out_dtype", "bfloat16")]
    out = torch.empty((s["M"], s["N"]), device=device, dtype=out_dt)
    out.fill_(_POISON)  # poison so partial/skipped writes are visible
    return out


def make_case(workload: dict, *, device: torch.device, seed: int):
    inputs = make_inputs(workload, device=device, seed=seed)
    return {
        "inputs": inputs,
        "baseline_outputs": {"out": _alloc_out(workload, device)},
        "candidate_outputs": {"out": _alloc_out(workload, device)},
        "tolerance": {"atol": workload.get("atol", 0.07), "rtol": workload.get("rtol", 0.02)},
    }


def call_baseline(workload: dict, inputs, outputs) -> None:
    build_ext.baseline(inputs["a"], inputs["b"], inputs["scale_a"], inputs["scale_b"], outputs["out"])


def call_candidate(workload: dict, inputs, outputs) -> None:
    build_ext.candidate(inputs["a"], inputs["b"], inputs["scale_a"], inputs["scale_b"], outputs["out"])


def compare_outputs(workload, baseline_outputs, candidate_outputs, tolerance) -> dict:
    b = baseline_outputs["out"].float()
    c = candidate_outputs["out"].float()
    if b.shape != c.shape or baseline_outputs["out"].dtype != candidate_outputs["out"].dtype:
        return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                "message": f"shape/dtype mismatch {tuple(b.shape)} vs {tuple(c.shape)}"}
    if torch.isnan(c).any() or torch.isinf(c).any():
        return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                "message": "candidate has NaN/Inf (poison not overwritten?)"}
    diff = (b - c).abs()
    max_abs = float(diff.max())
    max_rel = float((diff / (b.abs() + 1e-12)).max())
    ok = bool((diff <= tolerance["atol"] + tolerance["rtol"] * b.abs()).all())
    return {"ok": ok, "max_abs": max_abs, "max_rel": max_rel,
            "message": "" if ok else "candidate vs baseline exceeds tolerance"}
