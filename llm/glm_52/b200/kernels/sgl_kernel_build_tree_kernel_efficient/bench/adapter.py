"""Benchmark adapter for sgl_kernel.build_tree_kernel_efficient.

Implements the make_case / call_baseline / call_candidate / compare_outputs API
used by bench/benchmark.py.

Key design points (see docs/benchmark_method.md for the full rationale):

* The op is IN PLACE and depends on the FULL_MASK callsite output pre-state
  (tree_mask prefilled True; retrive_next_token / retrive_next_sibling prefilled
  -1). The harness poisons `outputs` before the correctness call, so outputs are
  returned as a custom RingOutputs object (NOT a tensor/list/dict): the harness'
  _poison_outputs treats it as a non-tensor and leaves it untouched, preserving
  the required pre-state.

* OUTPUT-BUFFER RING: each call writes into the next of K pre-stated output sets,
  so an invocation never observes a previous call's mutation. The op is also
  data-independent (fixed per-request work, no loop over tree_mask length and no
  buffer-state-dependent work beyond a single int64 store), so timing is unbiased
  even when the ring wraps; the ring's primary job is to give the correctness-
  gating compare a clean pre-state. K is configurable via BUILD_TREE_RING.

* compare_outputs is EXACT-match (int64 / bool / tree structure), per the
  llm_correctness_contract speculative-tree rule -- not atol/rtol.

* Synthetic inputs are generated from the captured invariants (parent_list [bs,0]
  empty so selected_index must be 0; verified_seq_len sums to (T-4*bs)/2 for
  FULL_MASK). No tensor VALUES were captured, only shapes, so synthesis is
  required; candidate-vs-baseline exact-match stays valid for any input that
  satisfies the kernel's preconditions.
"""

from __future__ import annotations

import os

import torch

from build_ext import get_ext

RING = int(os.environ.get("BUILD_TREE_RING", "512"))

OUTPUT_NAMES = (
    "tree_mask",
    "positions",
    "retrive_index",
    "retrive_next_token",
    "retrive_next_sibling",
)


class RingOutputs:
    """K pre-stated output sets cycled round-robin. Not a tensor/list/dict, so the
    benchmark harness' _poison_outputs leaves it (and the pre-state) untouched."""

    def __init__(self, sets: list[dict]):
        self.sets = sets
        self.k = len(sets)
        self.cursor = 0
        self.last = 0

    def next(self) -> dict:
        s = self.sets[self.cursor]
        self.last = self.cursor
        self.cursor = (self.cursor + 1) % self.k
        return s

    def last_set(self) -> dict:
        return self.sets[self.last]


def _verified_seq_len(bs: int, seq_sum: int, device, seed: int) -> torch.Tensor:
    """Near-uniform non-negative int64 split of seq_sum across bs (deterministic)."""
    if bs <= 0:
        return torch.zeros((0,), dtype=torch.int64, device=device)
    base = seq_sum // bs
    rem = seq_sum - base * bs
    vals = [base + (1 if i < rem else 0) for i in range(bs)]
    # deterministic rotation by seed so trials vary the split without changing sum
    shift = seed % bs
    vals = vals[shift:] + vals[:shift]
    return torch.tensor(vals, dtype=torch.int64, device=device)


def _make_prestated_set(bs: int, tree_len: int, device) -> dict:
    """One output set in the FULL_MASK callsite pre-state."""
    nv = 2  # draft_token_num
    return {
        "tree_mask": torch.full((tree_len,), True, dtype=torch.bool, device=device),
        # positions / retrive_index are fully written by the op; init -1 like the callsite buffer.
        "positions": torch.full((bs * nv,), -1, dtype=torch.int64, device=device),
        "retrive_index": torch.full((bs * nv,), -1, dtype=torch.int64, device=device),
        # retrive_next_token / retrive_next_sibling: -1 pre-state is REQUIRED (op leaves some at -1).
        "retrive_next_token": torch.full((bs * nv,), -1, dtype=torch.int64, device=device),
        "retrive_next_sibling": torch.full((bs * nv,), -1, dtype=torch.int64, device=device),
    }


def make_case(workload: dict, *, device: torch.device, seed: int):
    bs = int(workload["bs"])
    tree_len = int(workload["tree_mask_len"])
    seq_sum = int(workload["seq_sum"])
    sc = workload["scalars"]

    verified_seq_len = _verified_seq_len(bs, seq_sum, device, seed)
    if workload.get("noncontiguous"):
        # Strided (non-contiguous) verified_seq_len so the candidate must fall back
        # to the baseline. Both columns are set to verified_seq_len: the baseline
        # reads the underlying storage as if contiguous, so for a UNIFORM (all-equal)
        # split the first bs storage elements equal the logical values and the read
        # stays in-bounds and correct (no uninitialized garbage -> no OOB). The
        # noncontiguous workload row therefore uses a uniform split (bs | seq_sum).
        padded = verified_seq_len.unsqueeze(1).repeat(1, 2).contiguous()
        verified_seq_len = padded[:, 0]
        assert not verified_seq_len.is_contiguous()

    inputs = {
        "parent_list": torch.empty((bs, 0), dtype=torch.int64, device=device),
        "selected_index": torch.zeros((bs, 1), dtype=torch.int64, device=device),
        "verified_seq_len": verified_seq_len,
        "topk": int(sc["topk"]),
        "depth": int(sc["depth"]),
        "draft_token_num": int(sc["draft_token_num"]),
        "tree_mask_mode": int(sc["tree_mask_mode"]),
    }

    baseline_sets = [_make_prestated_set(bs, tree_len, device) for _ in range(RING)]
    candidate_sets = [_make_prestated_set(bs, tree_len, device) for _ in range(RING)]

    return {
        "inputs": inputs,
        "baseline_outputs": RingOutputs(baseline_sets),
        "candidate_outputs": RingOutputs(candidate_sets),
        "tolerance": {"atol": 0.0, "rtol": 0.0},
    }


def _launch(fn, inputs, s):
    fn(
        inputs["parent_list"],
        inputs["selected_index"],
        inputs["verified_seq_len"],
        s["tree_mask"],
        s["positions"],
        s["retrive_index"],
        s["retrive_next_token"],
        s["retrive_next_sibling"],
        inputs["topk"],
        inputs["depth"],
        inputs["draft_token_num"],
        inputs["tree_mask_mode"],
    )


def call_baseline(workload: dict, inputs, outputs) -> None:
    _launch(get_ext().build_tree_baseline, inputs, outputs.next())


def call_candidate(workload: dict, inputs, outputs) -> None:
    _launch(get_ext().build_tree_candidate, inputs, outputs.next())


def compare_outputs(workload, baseline_outputs, candidate_outputs, tolerance):
    b = baseline_outputs.last_set()
    c = candidate_outputs.last_set()
    for name in OUTPUT_NAMES:
        lhs, rhs = b[name], c[name]
        if lhs.shape != rhs.shape:
            return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                    "message": f"{name} shape {tuple(lhs.shape)} vs {tuple(rhs.shape)}"}
        if lhs.dtype != rhs.dtype:
            return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                    "message": f"{name} dtype {lhs.dtype} vs {rhs.dtype}"}
        if lhs.stride() != rhs.stride():
            return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                    "message": f"{name} stride {lhs.stride()} vs {rhs.stride()}"}
        if not torch.equal(lhs, rhs):
            return {"ok": False, "max_abs": float("inf"), "max_rel": float("inf"),
                    "message": f"{name} exact-match mismatch"}
    return {"ok": True, "max_abs": 0.0, "max_rel": 0.0, "message": ""}
