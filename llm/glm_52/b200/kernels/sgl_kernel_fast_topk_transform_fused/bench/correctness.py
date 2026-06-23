"""Correctness harness for `fast_topk_transform_fused` (GLM-5.2 / B200).

Runs the full frozen grid (production + regression rows in workloads.json) and
checks the candidate against the recovered baseline by EXACT integer match on
every output tensor, plus an independent PyTorch oracle on the unambiguous
length<=topk path. Output buffers are poisoned before each run so stale /
partial-write / skipped-kernel bugs are visible. `matched_ratio` must be 1.0
before any benchmark counts.

PROVISIONAL (recovery-and-scaffold round): runs once the task-local ABI is built
on the remote B200 (see bench/adapter.py). The exact valid-range mapping and the
returned-tensor count (one vs two; see docs/baseline_source.md) are confirmed by a
differential probe in the build round and folded in here before this gates a
benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import adapter  # bench/adapter.py

PAD_VALUE = adapter.PAD_VALUE  # -1
BENCH_DIR = Path(__file__).resolve().parent
WORKLOADS = BENCH_DIR / "workloads.json"


def reference_topk_transform_naive(score, lengths, page_table, cu_seqlens, topk):
    """Independent oracle for the length<=topk path (no real selection): each
    row takes its valid positions [0, length) in order, transforms via the
    sequence's page table, and pads the tail with -1. Tie-free and exact."""
    B = score.shape[0]
    out = torch.full((B, topk), PAD_VALUE, device=score.device, dtype=torch.int32)
    cu = cu_seqlens.tolist()
    seq_of_row = []
    for s in range(len(cu) - 1):
        seq_of_row.extend([s] * (cu[s + 1] - cu[s]))
    for b in range(B):
        s = seq_of_row[b] if b < len(seq_of_row) else len(cu) - 2
        length = int(lengths[s].item())
        n = min(length, topk)
        if n > 0:
            pos = torch.arange(n, device=score.device)
            out[b, :n] = page_table[s, pos].to(torch.int32)
    return out


def _checks(name, out, B, topk, device):
    msgs = []
    if tuple(out.shape) != (B, topk):
        msgs.append(f"{name} shape {tuple(out.shape)} != {(B, topk)}")
    if out.dtype != torch.int32:
        msgs.append(f"{name} dtype {out.dtype} != int32")
    if out.device.type != device.type:
        msgs.append(f"{name} device {out.device} != {device}")
    return msgs


def run(device_str: str = "cuda:0") -> int:
    device = torch.device(device_str)
    torch.set_grad_enabled(False)
    workloads = json.loads(WORKLOADS.read_text())
    total = 0
    passed = 0
    failures = []

    for wl in workloads:
        total += 1
        sc = wl["scalars"]
        B, topk = int(sc["B"]), int(sc["topk"])
        case = adapter.make_case(wl, device=device, seed=wl.get("seed", 0))

        # Poison destination buffers so partial / skipped writes are visible.
        for o in case.baseline_outputs + case.candidate_outputs:
            o.fill_(-17)

        adapter.call_baseline(wl, case.inputs, case.baseline_outputs)
        adapter.call_candidate(wl, case.inputs, case.candidate_outputs)
        torch.cuda.synchronize()

        problems = []
        for o in case.baseline_outputs:
            problems += _checks("baseline", o, B, topk, device)
        for o in case.candidate_outputs:
            problems += _checks("candidate", o, B, topk, device)

        cmp = adapter.compare_outputs(wl, case.baseline_outputs, case.candidate_outputs, case.tolerance)
        if not cmp["ok"]:
            problems.append(f"candidate!=baseline: {cmp['message']}")

        # Independent oracle cross-check on the unambiguous length<=topk path.
        if sc.get("lengths_mode", "full") in ("full", "half", "one", "zero") and int(sc["N"]) <= topk:
            ref = reference_topk_transform_naive(
                case.inputs["score"], case.inputs["lengths"],
                case.inputs["page_table_size_1"], case.inputs["cu_seqlens_q"], topk)
            if not torch.equal(ref, case.baseline_outputs[0]):
                mism = int((ref != case.baseline_outputs[0]).sum().item())
                problems.append(f"baseline!=oracle (naive path): {mism} mismatched entries")

        if problems:
            failures.append((wl["id"], problems))
        else:
            passed += 1

    ratio = passed / total if total else 0.0
    print(f"matched_ratio = {ratio:.4f}  ({passed}/{total} workloads)")
    for wid, probs in failures:
        print(f"  FAIL {wid}: {probs}")
    return 0 if ratio == 1.0 else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else "cuda:0"))
