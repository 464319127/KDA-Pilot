"""Benchmark/correctness adapter for `verify_tree_greedy`.

Implements the API documented in benchmark.py:
  make_case(workload, *, device, seed) -> dict
  call_baseline(workload, inputs, outputs) -> None
  call_candidate(workload, inputs, outputs) -> None
  compare_outputs(workload, baseline_outputs, candidate_outputs, tolerance) -> dict

Baseline and candidate are exposed by ONE shared torch CUDA extension
(`verify_tree_greedy_ext.cu`) with the identical outputs-last signature, built with
the same flags — see ../docs/benchmark_method.md. Outputs are exact integer/structural
tensors, so `compare_outputs` uses exact equality (atol/rtol do not apply).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch

import tree_inputs  # bench/ is on sys.path when benchmark.py imports the adapter

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# Input/output orderings for the shared ABI (inputs first, outputs last).
_INPUT_ORDER = (
    "candidates",
    "retrive_index",
    "retrive_next_token",
    "retrive_next_sibling",
    "target_predict",
)
_OUTPUT_ORDER = ("predicts", "accept_index", "accept_token_num")

_POISON = -17  # matches benchmark.py _poison_outputs for integer tensors

_ext = None
_ext_lock = threading.Lock()


def _get_ext():
    """Build (once) and return the shared baseline+candidate CUDA extension."""
    global _ext
    if _ext is None:
        with _ext_lock:
            if _ext is None:
                from torch.utils.cpp_extension import load

                _ext = load(
                    name="verify_tree_greedy_ext",
                    sources=[str(_HERE / "verify_tree_greedy_ext.cu")],
                    extra_include_paths=[str(_ROOT / "baseline"), str(_ROOT / "solution")],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3"],  # symmetric; no one-sided fast-math
                    verbose=False,
                )
    return _ext


def _shapes(workload: dict[str, Any]) -> tuple[int, int, int]:
    s = workload["shapes"]
    return int(s["bs"]), int(s["num_draft_tokens"]), int(s["num_spec_step"])


def make_case(workload: dict[str, Any], *, device: torch.device, seed: int) -> dict[str, Any]:
    bs, nd, nss = _shapes(workload)
    tree = workload.get("tree", {"mode": "random"})
    mode = tree.get("mode", "random")
    accept_prob = float(tree.get("accept_prob", 0.5))

    inp = tree_inputs.build_inputs(
        mode, bs, nd, nss, seed=seed, device=device, accept_prob=accept_prob
    )
    inputs = tuple(inp[k] for k in _INPUT_ORDER)

    base = tree_inputs.make_outputs(bs, nd, nss, poison=_POISON, device=device)
    cand = tree_inputs.make_outputs(bs, nd, nss, poison=_POISON, device=device)
    baseline_outputs = tuple(base[k] for k in _OUTPUT_ORDER)
    candidate_outputs = tuple(cand[k] for k in _OUTPUT_ORDER)

    return {
        "inputs": inputs,
        "baseline_outputs": baseline_outputs,
        "candidate_outputs": candidate_outputs,
        "tolerance": {"atol": 0.0, "rtol": 0.0},
    }


def call_baseline(workload: dict[str, Any], inputs, outputs) -> None:
    ext = _get_ext()
    ext.baseline_verify_tree_greedy(*inputs, *outputs)


def call_candidate(workload: dict[str, Any], inputs, outputs) -> None:
    ext = _get_ext()
    ext.candidate_verify_tree_greedy(*inputs, *outputs)


def compare_outputs(workload, baseline_outputs, candidate_outputs, tolerance) -> dict[str, Any]:
    """Exact integer/structural equality on all three outputs (incl. untouched slots)."""
    names = _OUTPUT_ORDER
    for name, b, c in zip(names, baseline_outputs, candidate_outputs):
        if b.shape != c.shape:
            return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                    "message": f"{name} shape mismatch: {tuple(b.shape)} vs {tuple(c.shape)}"}
        if b.dtype != c.dtype:
            return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                    "message": f"{name} dtype mismatch: {b.dtype} vs {c.dtype}"}
        if not torch.equal(b, c):
            diff = (b.to(torch.int64) - c.to(torch.int64)).abs()
            nmm = int((b != c).sum().item())
            return {"ok": False, "max_abs": float(diff.max().item()), "max_rel": 0.0,
                    "message": f"{name} differs in {nmm} element(s); "
                               f"baseline={b.flatten().tolist()[:16]} candidate={c.flatten().tolist()[:16]}"}
    return {"ok": True, "max_abs": 0.0, "max_rel": 0.0, "message": ""}
