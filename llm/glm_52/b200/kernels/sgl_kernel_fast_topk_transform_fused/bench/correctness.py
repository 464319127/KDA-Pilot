"""Correctness gate for `fast_topk_transform_fused` (GLM-5.2 / B200).

Runs the full frozen grid (production + regression rows in workloads.json) and checks the
candidate against the recovered baseline by EXACT integer match on every output, plus an
independent PyTorch oracle on the naive (length<=TopK) path. Output buffers are poisoned
before each run so stale / partial-write / skipped-kernel bugs are visible. `matched_ratio`
must be 1.0 before any benchmark counts.

Runs both as `python bench/correctness.py [device]` and `python -m bench.correctness`.
Requires the task-local ABI built on the remote B200 (see solution/build.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

try:                                   # python -m bench.correctness
    from . import adapter
except ImportError:                    # python bench/correctness.py
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import adapter

PAD_VALUE = adapter.PAD_VALUE  # -1
BENCH_DIR = Path(__file__).resolve().parent
WORKLOADS = BENCH_DIR / "workloads.json"
TOPK = 2048


def seq_index_per_row(B, cu_seqlens):
    """Map each token row b -> its sequence s, where cu[s] <= b < cu[s+1]."""
    cu = cu_seqlens.tolist()
    seq_of = [0] * B
    for s in range(len(cu) - 1):
        for b in range(cu[s], min(cu[s + 1], B)):
            seq_of[b] = s
    return seq_of


def reference_naive_transform(score, lengths, page_table, cu_seqlens, topk, is_decode):
    """Oracle for the naive path (every row's length <= topk): row b takes its valid
    positions [0, length_b) in order from its sequence's page table, pads tail with -1.
    Score-independent (so tie-insensitive); exact. Returns None if any row needs the
    radix path (length > topk), where exact tie-break is defined only by the kernel."""
    B = score.shape[0]
    lens = lengths.tolist()
    if any(L > topk for L in lens):
        return None
    out = torch.full((B, topk), PAD_VALUE, device=score.device, dtype=torch.int32)
    seq_of = list(range(B)) if is_decode else seq_index_per_row(B, cu_seqlens)
    for b in range(B):
        L = max(0, min(int(lens[b]), topk))
        if L > 0:
            out[b, :L] = page_table[seq_of[b], :L].to(torch.int32)
    return out


def _output_checks(name, out, B, topk, device):
    msgs = []
    if tuple(out.shape) != (B, topk):
        msgs.append(f"{name} shape {tuple(out.shape)} != {(B, topk)}")
    if out.dtype != torch.int32:
        msgs.append(f"{name} dtype {out.dtype} != int32")
    if out.device.type != device.type:
        msgs.append(f"{name} device {out.device.type} != {device.type}")
    if not out.is_contiguous():
        msgs.append(f"{name} not contiguous (stride {out.stride()})")
    if out.is_floating_point() and not torch.isfinite(out).all():
        msgs.append(f"{name} has NaN/Inf")
    return msgs


def run(device_str: str = "cuda:0") -> int:
    device = torch.device(device_str)
    torch.set_grad_enabled(False)
    workloads = json.loads(WORKLOADS.read_text())
    total = passed = 0
    failures = []

    for wl in workloads:
        total += 1
        sc = wl["scalars"]
        B, topk = int(sc["B"]), int(sc["topk"])
        is_decode = bool(sc.get("is_decode", sc["S"] == B))
        case = adapter.make_case(wl, device=device, seed=wl.get("seed", 0))

        # Poison destination buffers so partial/skipped writes are visible.
        for o in case.baseline_outputs + case.candidate_outputs:
            o.fill_(-17)

        adapter.call_baseline(wl, case.inputs, case.baseline_outputs)
        adapter.call_candidate(wl, case.inputs, case.candidate_outputs)
        torch.cuda.synchronize()

        problems = []
        for o in case.baseline_outputs:
            problems += _output_checks("baseline", o, B, topk, device)
        for o in case.candidate_outputs:
            problems += _output_checks("candidate", o, B, topk, device)

        cmp = adapter.compare_outputs(wl, case.baseline_outputs, case.candidate_outputs, case.tolerance)
        if not cmp["ok"]:
            problems.append(f"candidate != baseline: {cmp['message']}")

        ref = reference_naive_transform(
            case.inputs["score"], case.inputs["lengths"], case.inputs["page_table_size_1"],
            case.inputs["cu_seqlens_q"], topk, is_decode)
        if ref is not None and not torch.equal(ref, case.baseline_outputs[0]):
            mism = int((ref != case.baseline_outputs[0]).sum().item())
            problems.append(f"baseline != oracle (naive path): {mism} mismatched entries")

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
