"""Correctness gate for `fast_topk_transform_fused` (GLM-5.2 / B200).

Runs the full frozen grid (production + regression rows in workloads.json). The correctness
criterion is REGIME-SPLIT (the baseline radix path is non-deterministic in output order):
  - naive (length<=topk): exact candidate==baseline==independent-oracle (deterministic);
  - radix (length>topk): each output validated as a VALID top-k by full float key, tolerant of
    output order and exact-value ties (see validate_topk), honoring row_starts.
Output buffers are poisoned before each run so stale / partial-write / skipped-kernel bugs are
visible. `matched_ratio` must be 1.0 before any benchmark counts.

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


def validate_topk(name, out, inputs, topk, is_decode):
    """Radix path (length>topk): the baseline selects the TRUE top-k by full float key
    (`fast_topk_cuda_tl` refines the 8-bit boundary via 32-bit `convert_to_uint32` rounds),
    but the output ORDER (and exactly-equal-value ties) is race-nondeterministic. So validate
    each output as a VALID top-k independent of order:
      - invert each output page id through the ACTUAL `page_table[seq]` row (per-sequence scatter
        inverse — works for any unique page table, not just arange) to recover the selected
        relative position `pos` in [0, M);
      - require `pos` distinct, in [0, length), count == topk;
      - the radix reads `score[b, row_start + idx]` for idx in [0,length), so the selected scores
        are `score[b, row_start + pos]`; require their multiset == `torch.topk(score[b,
        row_start:row_start+length], topk).values` (sorted-equal, exact float)."""
    score = inputs["score"]
    lengths = inputs["lengths"]
    page_table = inputs["page_table_size_1"]
    cu = inputs["cu_seqlens_q"]
    row_starts = inputs.get("row_starts")
    B = score.shape[0]
    S, M = int(page_table.shape[0]), int(page_table.shape[1])
    seq_of = list(range(B)) if is_decode else seq_index_per_row(B, cu)
    # per-sequence inverse map: physical page id -> position in [0, M)
    maxv = int(page_table.max().item()) + 1
    inv = torch.full((S, maxv), -1, dtype=torch.long, device=score.device)
    cols = torch.arange(M, device=score.device, dtype=torch.long)
    for s in range(S):
        inv[s, page_table[s].long()] = cols
    for b in range(B):
        L = int(lengths[b])
        s = seq_of[b]
        rs = int(row_starts[b]) if row_starts is not None else 0
        ent = out[b].to(torch.int64)
        if not bool(((ent >= 0) & (ent < maxv)).all()):
            return False, f"{name} row {b}: output entry is not a valid page id"
        pos = inv[s, ent]  # recovered relative positions via the real page table
        if not bool(((pos >= 0) & (pos < L)).all()):
            return False, f"{name} row {b}: selected entry not in page_table[seq][:{L}]"
        if int(torch.unique(pos).numel()) != topk:
            return False, f"{name} row {b}: {int(torch.unique(pos).numel())} distinct positions != topk={topk}"
        sel_scores = score[b, rs + pos]
        true_top = torch.topk(score[b, rs:rs + L].float(), topk).values
        if not torch.equal(torch.sort(sel_scores.float()).values, torch.sort(true_top).values):
            return False, f"{name} row {b}: selected score multiset != true top-k"
    return True, ""


def _output_checks(name, out, B, topk, device):
    msgs = []
    if tuple(out.shape) != (B, topk):
        msgs.append(f"{name} shape {tuple(out.shape)} != {(B, topk)}")
    if out.dtype != torch.int32:
        msgs.append(f"{name} dtype {out.dtype} != int32")
    if out.device.type != device.type:
        msgs.append(f"{name} device {out.device.type} != {device.type}")
    # Exact device-index pin (GPU-pinning provenance): under CUDA_VISIBLE_DEVICES=1 the pinned GPU is
    # remapped to cuda:0, so the output must sit on that exact index, not just any CUDA device.
    if device.index is not None and out.device.index != device.index:
        msgs.append(f"{name} device index {out.device.index} != {device.index}")
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

        lengths = case.inputs["lengths"]
        is_radix = bool(lengths.numel()) and int(lengths.max().item()) > topk

        if not is_radix:
            # Naive path (length<=topk): deterministic identity+pad+transform.
            # Exact candidate-vs-baseline AND baseline-vs-independent-oracle.
            cmp = adapter.compare_outputs(wl, case.baseline_outputs, case.candidate_outputs, case.tolerance)
            if not cmp["ok"]:
                problems.append(f"candidate != baseline: {cmp['message']}")
            ref = reference_naive_transform(
                case.inputs["score"], lengths, case.inputs["page_table_size_1"],
                case.inputs["cu_seqlens_q"], topk, is_decode)
            if ref is not None and not torch.equal(ref, case.baseline_outputs[0]):
                mism = int((ref != case.baseline_outputs[0]).sum().item())
                problems.append(f"baseline != oracle (naive path): {mism} mismatched entries")
        else:
            # Radix path (length>topk): baseline output ORDER is race-nondeterministic, so do
            # NOT exact-compare candidate vs baseline. Instead validate EACH output as a valid
            # top-k by full float key (order/tie tolerant).
            for nm, out in (("baseline", case.baseline_outputs[0]), ("candidate", case.candidate_outputs[0])):
                ok, msg = validate_topk(nm, out, case.inputs, topk, is_decode)
                if not ok:
                    problems.append(msg)

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
