#!/usr/bin/env python3
"""Correctness suite for build_tree_kernel_efficient (baseline + native-CUDA candidate).

Per the llm_correctness_contract speculative-tree rule, integer/bool/tree-structure
outputs are EXACT-match. Coverage:

* INDEPENDENT ORACLE for the captured fixed regime (topk=1, depth=1,
  draft_token_num=2, FULL_MASK, selected_index==0), computed in pure Python from
  the recovered semantics — both baseline and candidate are checked against it.
* POISON: positions / retrive_index (fully written) pre-filled with a sentinel to
  catch partial writes; tree_mask / retrive_next_token / retrive_next_sibling carry
  the REQUIRED callsite pre-state (True / -1 / -1).
* IN-PLACE: baseline and candidate run on SEPARATE copies; compared to the oracle
  and to each other, with shape/dtype/device/stride checks.
* COVERAGE: every production row (multiple verified_seq_len distributions), the full
  captured (bs,T) range sweep, and the baseline-FALLBACK domain — non-FULL_MASK
  mode, non-contiguous verified_seq_len, draft_token_num!=2, wrong parent_list
  dtype, and wrong selected_index shape — where the candidate must reproduce the
  baseline bit-for-bit (its dispatch must route off-domain inputs to the baseline).

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
import build_ext  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POISON = -17
NV = 2  # draft_token_num for the fixed regime


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
        w = list(range(1, bs + 1))
        sw = sum(w)
        v = [seq_sum * x // sw for x in w]
        v[-1] += seq_sum - sum(v)
        return v
    cuts = sorted(torch.randint(0, seq_sum + 1, (bs - 1,), generator=g).tolist())
    pts = [0] + cuts + [seq_sum]
    return [pts[i + 1] - pts[i] for i in range(bs)]


def oracle_full_mask(L: list[int], device) -> dict:
    """Pure-Python expected post-state for the fixed regime (selected_index==0)."""
    bs = len(L)
    T = 2 * sum(L) + 4 * bs
    tm = torch.ones(T, dtype=torch.bool, device=device)
    pos = torch.empty(bs * NV, dtype=torch.int64, device=device)       # 1-D [bs*NV]
    ri = torch.empty((bs, NV), dtype=torch.int64, device=device)       # 2-D [bs, NV]
    rnt = torch.empty((bs, NV), dtype=torch.int64, device=device)      # 2-D [bs, NV]
    rns = torch.empty((bs, NV), dtype=torch.int64, device=device)      # 2-D [bs, NV]
    prefix = 0
    for b in range(bs):
        Lb = L[b]
        S = 4 * b + 2 * prefix
        tm[S + Lb + 1] = False
        pos[2 * b], pos[2 * b + 1] = Lb, Lb + 1
        ri[b, 0], ri[b, 1] = 2 * b, 2 * b + 1
        rnt[b, 0], rnt[b, 1] = 1, -1
        rns[b, 0], rns[b, 1] = -1, -1
        prefix += Lb
    return {"tree_mask": tm, "positions": pos, "retrive_index": ri,
            "retrive_next_token": rnt, "retrive_next_sibling": rns}


def prestated_outputs(total_tree: int, bs: int, nv: int, device) -> dict:
    """Required callsite pre-state in the captured shapes: positions [bs*nv] (1-D),
    retrive_* [bs, nv] (2-D). positions/retrive_index POISONED to catch partial writes."""
    return {
        "tree_mask": torch.full((total_tree,), True, dtype=torch.bool, device=device),
        "positions": torch.full((bs * nv,), POISON, dtype=torch.int64, device=device),
        "retrive_index": torch.full((bs, nv), POISON, dtype=torch.int64, device=device),
        "retrive_next_token": torch.full((bs, nv), -1, dtype=torch.int64, device=device),
        "retrive_next_sibling": torch.full((bs, nv), -1, dtype=torch.int64, device=device),
    }


def run_op(fn, parent_list, selected_index, vsl, out, sc):
    fn(parent_list, selected_index, vsl, out["tree_mask"], out["positions"],
       out["retrive_index"], out["retrive_next_token"], out["retrive_next_sibling"],
       sc["topk"], sc["depth"], sc["draft_token_num"], sc["tree_mask_mode"])


def route_of(parent_list, selected_index, vsl, out, sc) -> int:
    """1 = candidate native fast path, 0 = baseline fallback (no kernel launch)."""
    return build_ext.route(
        parent_list, selected_index, vsl, out["tree_mask"], out["positions"],
        out["retrive_index"], out["retrive_next_token"], out["retrive_next_sibling"],
        sc["topk"], sc["depth"], sc["draft_token_num"], sc["tree_mask_mode"])


def exact_eq(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride() and torch.equal(a, b)


# ----------------------------- FULL_MASK fixed regime -----------------------------
def check_full_mask(device, bs, L, label, fails):
    T = 2 * sum(L) + 4 * bs
    parent_list = torch.empty((bs, 0), dtype=torch.int64, device=device)
    selected_index = torch.zeros((bs, 1), dtype=torch.int64, device=device)
    vsl = torch.tensor(L, dtype=torch.int64, device=device)
    sc = {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": 0}

    base_out = prestated_outputs(T, bs, NV, device)
    cand_out = prestated_outputs(T, bs, NV, device)  # SEPARATE copy
    # PROVE the candidate actually takes the native fast path for this captured
    # production regime (route==1) — not a silent baseline fallback.
    if route_of(parent_list, selected_index, vsl, cand_out, sc) != 1:
        fails.append(f"[{label}] candidate did NOT take the native fast path (route!=1)")
    run_op(build_ext.baseline, parent_list, selected_index, vsl, base_out, sc)
    run_op(build_ext.candidate, parent_list, selected_index, vsl, cand_out, sc)
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


# ----------------------------- fallback domain (candidate must == baseline) -----------------------------
def check_identity(device, label, parent_list, selected_index, vsl, sc, total_tree, total_idx, fails):
    nv = int(sc["draft_token_num"])
    cbs = int(parent_list.shape[0])  # total_idx == cbs * nv
    base_out = prestated_outputs(total_tree, cbs, nv, device)
    cand_out = prestated_outputs(total_tree, cbs, nv, device)
    # PROVE this off-domain case routes to the baseline fallback (route==0).
    if route_of(parent_list, selected_index, vsl, cand_out, sc) != 0:
        fails.append(f"[{label}] expected baseline fallback but took fast path (route!=0)")
    run_op(build_ext.baseline, parent_list, selected_index, vsl, base_out, sc)
    run_op(build_ext.candidate, parent_list, selected_index, vsl, cand_out, sc)
    torch.cuda.synchronize()
    for name in base_out:
        if not exact_eq(cand_out[name], base_out[name]):
            fails.append(f"[{label}] fallback candidate {name} != baseline")


def fallback_cases(device, fails):
    # 1) non-FULL_MASK mode (QLEN_ONLY): tree_mask size = draft^2 * bs
    bs = 4
    vsl = torch.tensor([5, 7, 3, 9], dtype=torch.int64, device=device)
    pl = torch.empty((bs, 0), dtype=torch.int64, device=device)
    si = torch.zeros((bs, 1), dtype=torch.int64, device=device)
    check_identity(device, "fallback/qlen_only", pl, si, vsl,
                   {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": 1},
                   NV * NV * bs, bs * NV, fails)

    # 2) non-contiguous verified_seq_len
    base = torch.tensor([5, 7, 3, 9], dtype=torch.int64, device=device).unsqueeze(1).repeat(1, 2).contiguous()
    vsl_nc = base[:, 0]
    assert not vsl_nc.is_contiguous()
    T = 2 * int(vsl_nc.sum()) + 4 * bs
    check_identity(device, "fallback/noncontig_vsl", pl, si, vsl_nc,
                   {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": 0},
                   T, bs * NV, fails)

    # 3) wrong parent_list dtype (int32) -> dtype guard -> fallback (parent_list [bs,0] never read)
    pl_i32 = torch.empty((bs, 0), dtype=torch.int32, device=device)
    check_identity(device, "fallback/parentlist_int32", pl_i32, si, vsl,
                   {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": 0},
                   2 * int(vsl.sum()) + 4 * bs, bs * NV, fails)

    # 4) wrong selected_index shape [bs,2] (expected [bs,1]) -> shape guard -> fallback
    si_bad = torch.zeros((bs, 2), dtype=torch.int64, device=device)
    check_identity(device, "fallback/selidx_shape", pl, si_bad, vsl,
                   {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": 0},
                   2 * int(vsl.sum()) + 4 * bs, bs * NV, fails)

    # 5) draft_token_num=4 -> scalar guard -> fallback (depth=1 -> parent_list [bs,1], selected_index 0)
    d = 4
    pl4 = torch.zeros((bs, 1), dtype=torch.int64, device=device)
    si4 = torch.zeros((bs, d - 1), dtype=torch.int64, device=device)
    T4 = int(vsl.sum()) * d + d * d * bs
    check_identity(device, "fallback/draft4", pl4, si4, vsl,
                   {"topk": 1, "depth": 1, "draft_token_num": d, "tree_mask_mode": 0},
                   T4, bs * d, fails)


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
    build_ext.get_ext()
    fails: list[str] = []
    n = 0

    rows = json.loads((HERE / "workloads.json").read_text())
    for row in rows:
        if not row["production"]:
            continue  # fallback/edge rows handled explicitly below
        bs = int(row["bs"])
        dists = ("uniform",) if bs == 1 else ("uniform", "skewed", "monotonic", "random")
        for dist in dists:
            L = seq_lens(bs, int(row["seq_sum"]), dist, 1234 + bs)
            check_full_mask(device, bs, L, f"{row['id']}/{dist}", fails)
            n += 1

    for bs, seq_sum in evidence_bsT_sweep():
        L = seq_lens(bs, seq_sum, "uniform", 7)
        check_full_mask(device, bs, L, f"sweep_bs{bs}_seqsum{seq_sum}", fails)
        n += 1

    fallback_cases(device, fails)
    n += 5

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
