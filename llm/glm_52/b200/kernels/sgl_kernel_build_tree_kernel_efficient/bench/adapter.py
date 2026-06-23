"""Benchmark adapter for sgl_kernel.build_tree_kernel_efficient.

Implements make_case / call_baseline / call_candidate / compare_outputs for
bench/benchmark.py.

Pre-state contract (AC-6): the op is IN PLACE and depends on the FULL_MASK
callsite output pre-state (tree_mask prefilled True; retrive_next_token /
retrive_next_sibling prefilled -1). The harness poisons `outputs` before the
correctness call, so outputs are returned as a custom RingOutputs object (NOT a
tensor/list/dict) — the harness' _poison_outputs treats it as a non-tensor and
leaves the pre-state untouched.

OUTPUT-BUFFER RING (no wrap): each invocation writes into a fresh, pre-stated
output set so it NEVER observes a prior call's mutation. The ring is held as
contiguous 2D tensors (RING x elems) whose rows are the per-call output views, so
memory + allocation stay cheap regardless of RING. RING (default 16384) is sized
to exceed the maximum per-trial invocation count for the configured benchmark
(correctness 1 + warmup + calibration sum<=8191 + timed<=inner_max=4096 ~= 12298
for inner_max=4096); RingOutputs.next() HARD-ASSERTS it never wraps. make_case is
called per trial, so the cursor resets each trial.

Synthetic inputs come from the captured invariants (parent_list [bs,0] empty so
selected_index must be 0; verified_seq_len sums to (T-4*bs)/2 for FULL_MASK). No
tensor VALUES were captured, only shapes; candidate-vs-baseline exact-match stays
valid for any input satisfying the kernel's preconditions.
"""

from __future__ import annotations

import os

import torch

import build_ext

RING = int(os.environ.get("BUILD_TREE_RING", "16384"))
NV = 2  # draft_token_num for the captured fixed regime

OUTPUT_NAMES = (
    "tree_mask",
    "positions",
    "retrive_index",
    "retrive_next_token",
    "retrive_next_sibling",
)


class RingOutputs:
    """RING pre-stated output sets as 2D tensors; next() returns row-views and
    hard-asserts it never wraps onto an already-used (mutated) set."""

    def __init__(self, rings: dict, k: int):
        self._rings = rings  # name -> 2D tensor (k, elems)
        self.k = k
        self.cursor = 0
        self.last = 0

    def next(self) -> dict:
        if self.cursor >= self.k:
            raise RuntimeError(
                f"output-buffer ring wrapped (k={self.k}); a measured invocation would "
                f"reuse a mutated set. Increase BUILD_TREE_RING."
            )
        i = self.cursor
        self.last = i
        self.cursor += 1
        return {name: t[i] for name, t in self._rings.items()}

    def last_set(self) -> dict:
        return {name: t[self.last] for name, t in self._rings.items()}


def _make_ring(bs: int, tree_len: int, device, k: int) -> dict:
    """k pre-stated output sets (FULL_MASK pre-state) as contiguous 2D tensors."""
    rings = {
        "tree_mask": torch.empty((k, tree_len), dtype=torch.bool, device=device),
        "positions": torch.empty((k, bs * NV), dtype=torch.int64, device=device),
        "retrive_index": torch.empty((k, bs * NV), dtype=torch.int64, device=device),
        "retrive_next_token": torch.empty((k, bs * NV), dtype=torch.int64, device=device),
        "retrive_next_sibling": torch.empty((k, bs * NV), dtype=torch.int64, device=device),
    }
    rings["tree_mask"].fill_(True)
    for name in ("positions", "retrive_index", "retrive_next_token", "retrive_next_sibling"):
        rings[name].fill_(-1)
    return rings


def _verified_seq_len(bs: int, seq_sum: int, device, seed: int) -> torch.Tensor:
    """Near-uniform non-negative int64 split of seq_sum across bs (deterministic)."""
    if bs <= 0:
        return torch.zeros((0,), dtype=torch.int64, device=device)
    base = seq_sum // bs
    rem = seq_sum - base * bs
    vals = [base + (1 if i < rem else 0) for i in range(bs)]
    shift = seed % bs
    vals = vals[shift:] + vals[:shift]
    return torch.tensor(vals, dtype=torch.int64, device=device)


def make_case(workload: dict, *, device: torch.device, seed: int):
    bs = int(workload["bs"])
    tree_len = int(workload["tree_mask_len"])
    seq_sum = int(workload["seq_sum"])
    sc = workload["scalars"]

    verified_seq_len = _verified_seq_len(bs, seq_sum, device, seed)
    if workload.get("noncontiguous") == "verified_seq_len":
        # Strided (non-contiguous) verified_seq_len so the candidate must fall back
        # to baseline. Both columns hold the value (uniform split) so the
        # contiguous-assuming baseline pointer read stays in-bounds.
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

    return {
        "inputs": inputs,
        "baseline_outputs": RingOutputs(_make_ring(bs, tree_len, device, RING), RING),
        "candidate_outputs": RingOutputs(_make_ring(bs, tree_len, device, RING), RING),
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
    _launch(build_ext.baseline, inputs, outputs.next())


def call_candidate(workload: dict, inputs, outputs) -> None:
    _launch(build_ext.candidate, inputs, outputs.next())


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
