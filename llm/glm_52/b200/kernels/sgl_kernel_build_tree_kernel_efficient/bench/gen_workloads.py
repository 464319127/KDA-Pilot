#!/usr/bin/env python3
"""Freeze bench/workloads.json from docs/evidence.json (run once, before tuning).

The captured interface has FIXED scalars across all 187 variants
(topk=1, depth=1, draft_token_num=2, tree_mask_mode=FULL_MASK). The only varying
dimensions are batch size `bs` and the bool tree_mask length `T`, related by the
invariant T = 2*sum(verified_seq_len) + 4*bs (verified: 0 violations over 187).

The op's work is determined by `bs` and is INDEPENDENT of `T` (it writes a fixed
number of entries per request and loops only over prior requests, never over T or
seq_len). Therefore the 183 distinct (bs, T) captured shapes collapse, for the
performance HEADLINE, into one representative production row per distinct bs
(1..10), each pinned to a real captured (bs, T) shape (the median captured T for
that bs). This is an explicit, documented collapse -- not a silent drop: every
captured bs bucket is represented, and bench/correctness.py separately sweeps the
full captured (bs, T) range and multiple seq-length distributions.

Regression-only edge rows (production:false, excluded from the headline) cover the
extreme tree_mask lengths and the baseline-fallback path (a non-FULL_MASK mode).
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVIDENCE = ROOT / "docs" / "evidence.json"
OUT = HERE / "workloads.json"

ARG0 = re.compile(r"arg\[0\]=Tensor\(\s*shape=\((\d+), \d+\)")
ARG3 = re.compile(r"arg\[3\]=Tensor\(\s*shape=\((\d+),\)")


def main() -> None:
    ev = json.loads(EVIDENCE.read_text())
    counts: collections.Counter = collections.Counter()
    per_bs = collections.defaultdict(list)
    for v in ev["variants"]:
        raw = v["args"][0]["raw"]
        bs = int(ARG0.search(raw).group(1))
        T = int(ARG3.search(raw).group(1))
        rem = T - 4 * bs
        assert rem >= 0 and rem % 2 == 0, f"invariant violation bs={bs} T={T}"
        counts[(bs, T)] += v.get("call_count", 1)
        per_bs[bs].append(T)

    rows = []

    # Production / headline rows: one representative real captured shape per bs.
    for bs in sorted(per_bs):
        ts = sorted(per_bs[bs])
        T = ts[len(ts) // 2]  # median captured T for this bs (a real captured shape)
        seq_sum = (T - 4 * bs) // 2
        captured_calls = sum(c for (b, _), c in counts.items() if b == bs)
        rows.append(
            {
                "id": f"glm52_bs{bs}_T{T}",
                "production": True,
                "function": "build_tree",
                "bs": bs,
                "tree_mask_len": T,
                "seq_sum": seq_sum,
                "dtypes": {"int": "int64", "tree_mask": "bool"},
                "scalars": {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": 0},
                "distinct_T_for_bs": len(set(per_bs[bs])),
                "captured_calls_for_bs": captured_calls,
                "atol": 0.0,
                "rtol": 0.0,
                "seed": 0,
            }
        )

    # Regression-only edge rows (excluded from the headline geomean).
    bs_min, T_min = min(counts, key=lambda k: k[1])
    bs_max, T_max = max(counts, key=lambda k: k[1])
    edge = [
        ("edge_minT", bs_min, T_min, 0),     # smallest captured tree_mask
        ("edge_maxT", bs_max, T_max, 0),     # largest captured tree_mask
        ("edge_bs1_seq0", 1, 4, 0),          # degenerate: seq_len=0 -> T=4
    ]
    for name, bs, T, mode in edge:
        rows.append(
            {
                "id": name,
                "production": False,
                "function": "build_tree",
                "bs": bs,
                "tree_mask_len": T,
                "seq_sum": (T - 4 * bs) // 2,
                "dtypes": {"int": "int64", "tree_mask": "bool"},
                "scalars": {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": mode},
                "atol": 0.0,
                "rtol": 0.0,
                "seed": 0,
            }
        )

    # Fallback regression row: QLEN_ONLY mode (1) -> candidate must route to baseline.
    # QLEN_ONLY tree_mask length = num_verify*num_verify*bs = 4*bs.
    rows.append(
        {
            "id": "fallback_qlen_only_bs4",
            "production": False,
            "function": "build_tree",
            "bs": 4,
            "tree_mask_len": 4 * 4,  # 4*bs, bs=4
            "seq_sum": 24,           # arbitrary valid; QLEN_ONLY tree_mask size ignores seq_sum
            "dtypes": {"int": "int64", "tree_mask": "bool"},
            "scalars": {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": 1},
            "fallback_expected": True,
            "atol": 0.0,
            "rtol": 0.0,
            "seed": 0,
        }
    )

    # Fallback regression row: non-contiguous verified_seq_len -> candidate routes to baseline.
    rows.append(
        {
            "id": "fallback_noncontig_bs4",
            "production": False,
            "function": "build_tree",
            "bs": 4,
            "tree_mask_len": (24) * 2 + 4 * 4,
            "seq_sum": 24,
            "dtypes": {"int": "int64", "tree_mask": "bool"},
            "scalars": {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": 0},
            "noncontiguous": True,
            "fallback_expected": True,
            "atol": 0.0,
            "rtol": 0.0,
            "seed": 0,
        }
    )

    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    prod = sum(1 for r in rows if r["production"])
    print(f"wrote {OUT} : {len(rows)} rows ({prod} production/headline, {len(rows)-prod} regression)")


if __name__ == "__main__":
    main()
