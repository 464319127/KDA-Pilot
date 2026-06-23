#!/usr/bin/env python3
"""Correctness suite for build_tree_kernel_efficient (baseline + native-CUDA candidate).

Checks, per the llm_correctness_contract speculative-tree rule (EXACT match for
integer / bool / tree-structure outputs):

* INDEPENDENT ORACLE: the expected post-state for the captured fixed regime
  (topk=1, depth=1, draft_token_num=2, FULL_MASK, selected_index==0) is computed
  in pure Python from the recovered semantics, NOT from the CUDA baseline. Both
  baseline and candidate are checked against it.
* POISON: positions / retrive_index (fully written by the op) are pre-filled with
  a sentinel so a partial / skipped write is detected; tree_mask /
  retrive_next_token / retrive_next_sibling carry the REQUIRED callsite pre-state
  (True / -1 / -1).
* IN-PLACE: baseline and candidate run on SEPARATE copies; we compare each to the
  oracle and to each other, plus shape / dtype / device / stride.
* COVERAGE: every workloads.json row; multiple verified_seq_len distributions
  (uniform / skewed / monotonic / random) for multi-batch rows; a sweep over the
  full captured (bs, T) range (per-bs min / median / max T from evidence.json);
  and the baseline-fallback rows (non-FULL_MASK mode and non-contiguous input),
  where the candidate must reproduce the baseline bit-for-bit.

Run: python bench/correctness.py   (exit 0 iff all cases pass)
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ext import get_ext  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POISON = -17
NV = 2  # draft_token_num


# ----------------------------- input construction -----------------------------
def seq_lens(bs: int, seq_sum: int, dist: str, seed: int) -> list[int]:
    if bs == 1:
        return [seq_sum]
    g = torch.Generator().manual_seed(seed)
    if dist == "uniform":
        base = seq_sum // bs
        rem = seq_sum - base * bs
        return [base + (1 if i < rem else 0) for i in range(bs)]
    if dist == "skewed":
        v = [0] * bs
        v[0] = seq_sum
        return v
    if dist == "monotonic":
        # increasing, summing to seq_sum
        w = list(range(1, bs + 1))
        sw = sum(w)
        v = [seq_sum * x // sw for x in w]
        v[-1] += seq_sum - sum(v)
        return v
    # random non-negative split
    cuts = sorted(torch.randint(0, seq_sum + 1, (bs - 1,), generator=g).tolist())
    pts = [0] + cuts + [seq_sum]
    return [pts[i + 1] - pts[i] for i in range(bs)]


def oracle_full_mask(L: list[int], device) -> dict:
    """Pure-Python expected post-state for the fixed regime (selected_index==0)."""
    bs = len(L)
    T = 2 * sum(L) + 4 * bs
    tm = torch.ones(T, dtype=torch.bool, device=device)
    pos = torch.empty(bs * NV, dtype=torch.int64, device=device)
    ri = torch.empty(bs * NV, dtype=torch.int64, device=device)
    rnt = torch.empty(bs * NV, dtype=torch.int64, device=device)
    rns = torch.empty(bs * NV, dtype=torch.int64, device=device)
    prefix = 0
    for b in range(bs):
        Lb = L[b]
        S = 4 * b + 2 * prefix
        tm[S + Lb + 1] = False
        pos[2 * b], pos[2 * b + 1] = Lb, Lb + 1
        ri[2 * b], ri[2 * b + 1] = 2 * b, 2 * b + 1
        rnt[2 * b], rnt[2 * b + 1] = 1, -1
        rns[2 * b], rns[2 * b + 1] = -1, -1
        prefix += Lb
    return {"tree_mask": tm, "positions": pos, "retrive_index": ri,
            "retrive_next_token": rnt, "retrive_next_sibling": rns}


def prestated_outputs(bs: int, T: int, device) -> dict:
    """Required callsite pre-state, with positions/retrive_index POISONED to catch
    partial writes."""
    return {
        "tree_mask": torch.full((T,), True, dtype=torch.bool, device=device),
        "positions": torch.full((bs * NV,), POISON, dtype=torch.int64, device=device),
        "retrive_index": torch.full((bs * NV,), POISON, dtype=torch.int64, device=device),
        "retrive_next_token": torch.full((bs * NV,), -1, dtype=torch.int64, device=device),
        "retrive_next_sibling": torch.full((bs * NV,), -1, dtype=torch.int64, device=device),
    }


def run_op(fn, parent_list, selected_index, vsl, out, scalars):
    fn(parent_list, selected_index, vsl, out["tree_mask"], out["positions"],
       out["retrive_index"], out["retrive_next_token"], out["retrive_next_sibling"],
       scalars["topk"], scalars["depth"], scalars["draft_token_num"], scalars["tree_mask_mode"])


def exact_eq(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride() and torch.equal(a, b)


# ----------------------------- one FULL_MASK case -----------------------------
def check_full_mask(ext, device, bs, L, label, fails):
    seq_sum = sum(L)
    T = 2 * seq_sum + 4 * bs
    parent_list = torch.empty((bs, 0), dtype=torch.int64, device=device)
    selected_index = torch.zeros((bs, 1), dtype=torch.int64, device=device)
    vsl = torch.tensor(L, dtype=torch.int64, device=device)
    sc = {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": 0}

    base_out = prestated_outputs(bs, T, device)
    cand_out = prestated_outputs(bs, T, device)  # SEPARATE copy
    run_op(ext.build_tree_baseline, parent_list, selected_index, vsl, base_out, sc)
    run_op(ext.build_tree_candidate, parent_list, selected_index, vsl, cand_out, sc)
    torch.cuda.synchronize()

    oracle = oracle_full_mask(L, device)
    for name in oracle:
        if not exact_eq(base_out[name], oracle[name]):
            fails.append(f"[{label}] baseline {name} != oracle")
        if not exact_eq(cand_out[name], oracle[name]):
            fails.append(f"[{label}] candidate {name} != oracle")
        if not exact_eq(cand_out[name], base_out[name]):
            fails.append(f"[{label}] candidate {name} != baseline")
        if (base_out[name] == POISON).any() or (cand_out[name] == POISON).any():
            fails.append(f"[{label}] {name} left POISON (partial write)")


def check_fallback(ext, device, row, fails):
    """Non-FULL_MASK or non-contiguous: candidate must equal baseline exactly."""
    bs = int(row["bs"])
    sc = {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": int(row["scalars"]["tree_mask_mode"])}
    label = row["id"]
    seq_sum = int(row["seq_sum"])
    L = seq_lens(bs, seq_sum, "uniform", 0)
    vsl = torch.tensor(L, dtype=torch.int64, device=device)
    if row.get("noncontiguous"):
        # Both columns = vsl so the baseline's contiguous storage read stays
        # in-bounds for a uniform split (see adapter.make_case for rationale).
        padded = vsl.unsqueeze(1).repeat(1, 2).contiguous()
        vsl = padded[:, 0]
        if vsl.is_contiguous():
            fails.append(f"[{label}] expected non-contiguous verified_seq_len")
    parent_list = torch.empty((bs, 0), dtype=torch.int64, device=device)
    selected_index = torch.zeros((bs, 1), dtype=torch.int64, device=device)
    T = int(row["tree_mask_len"])
    base_out = prestated_outputs(bs, T, device)
    cand_out = prestated_outputs(bs, T, device)
    run_op(ext.build_tree_baseline, parent_list, selected_index, vsl, base_out, sc)
    run_op(ext.build_tree_candidate, parent_list, selected_index, vsl, cand_out, sc)
    torch.cuda.synchronize()
    for name in base_out:
        if not exact_eq(cand_out[name], base_out[name]):
            fails.append(f"[{label}] fallback candidate {name} != baseline")


def evidence_bsT_sweep():
    """Per-bs (min, median, max) T from the captured evidence, for range coverage."""
    ev = json.loads((ROOT / "docs" / "evidence.json").read_text())
    a0 = re.compile(r"arg\[0\]=Tensor\(\s*shape=\((\d+), \d+\)")
    a3 = re.compile(r"arg\[3\]=Tensor\(\s*shape=\((\d+),\)")
    per = collections.defaultdict(set)
    for v in ev["variants"]:
        raw = v["args"][0]["raw"]
        per[int(a0.search(raw).group(1))].add(int(a3.search(raw).group(1)))
    out = []
    for bs, ts in sorted(per.items()):
        ts = sorted(ts)
        for T in {ts[0], ts[len(ts) // 2], ts[-1]}:
            out.append((bs, (T - 4 * bs) // 2))
    return out


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    device = torch.device("cuda")
    ext = get_ext()
    fails: list[str] = []
    n = 0

    rows = json.loads((HERE / "workloads.json").read_text())
    for row in rows:
        bs = int(row["bs"])
        if row.get("fallback_expected"):
            check_fallback(ext, device, row, fails)
            n += 1
            continue
        for dist in ("uniform", "skewed", "monotonic", "random"):
            L = seq_lens(bs, int(row["seq_sum"]), dist, 1234 + bs)
            check_full_mask(ext, device, bs, L, f"{row['id']}/{dist}", fails)
            n += 1

    # full captured (bs, T) range sweep
    for bs, seq_sum in evidence_bsT_sweep():
        L = seq_lens(bs, seq_sum, "uniform", 7)
        check_full_mask(ext, device, bs, L, f"sweep_bs{bs}_seqsum{seq_sum}", fails)
        n += 1

    print(f"ran {n} cases; {len(fails)} failures")
    for f in fails[:50]:
        print("  FAIL", f)
    if fails:
        print("CORRECTNESS: FAIL")
        return 1
    print("CORRECTNESS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
