#!/usr/bin/env python3
"""Freeze bench/workloads.json from docs/evidence.json (run once, before tuning).

AC-3: EVERY distinct captured production (bs, T, dtype, stride, scalar) combination
is emitted as a production row — no captured bucket is dropped. The captured
interface has FIXED scalars across all 187 variants (topk=1, depth=1,
draft_token_num=2, tree_mask_mode=FULL_MASK) and all tensors are contiguous
(is_contiguous=True in the evidence); the distinguishing shape axes are batch size
`bs` and the bool tree_mask length `T`, related by T = 2*sum(verified_seq_len) +
4*bs (verified: 0 violations over 187 variants).

Production rows = all distinct captured (bs, T) (183 rows). Regression-only rows
(production:false, excluded from the headline geomean) cover synthetic edges and
the baseline-fallback domain (degenerate seq, non-FULL_MASK mode, non-contiguous
inputs, off-domain dtype/shape) so the candidate dispatch guards are exercised.
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

# All captured tensors are contiguous (row-major). Strides recorded per AC-3.
CONTIG = "contiguous (row-major; all captured variants is_contiguous=True)"
DTYPES = {"int": "int64", "tree_mask": "bool"}
FULL_MASK = {"topk": 1, "depth": 1, "draft_token_num": 2, "tree_mask_mode": 0}


def base_row(rid, bs, T, scalars, production, **extra):
    return {
        "id": rid,
        "production": production,
        "function": "build_tree",
        "bs": bs,
        "tree_mask_len": T,
        "seq_sum": (T - 4 * bs) // 2,
        "dtypes": DTYPES,
        "strides": CONTIG,
        "contiguous": True,
        "scalars": scalars,
        "atol": 0.0,
        "rtol": 0.0,
        "seed": 0,
        **extra,
    }


def main() -> None:
    ev = json.loads(EVIDENCE.read_text())
    counts: collections.Counter = collections.Counter()
    for v in ev["variants"]:
        raw = v["args"][0]["raw"]
        bs = int(ARG0.search(raw).group(1))
        T = int(ARG3.search(raw).group(1))
        rem = T - 4 * bs
        assert rem >= 0 and rem % 2 == 0, f"invariant violation bs={bs} T={T}"
        counts[(bs, T)] += v.get("call_count", 1)

    rows = []
    # Production rows: every distinct captured (bs, T), sorted for stability.
    for (bs, T) in sorted(counts):
        rows.append(
            base_row(f"glm52_bs{bs}_T{T}", bs, T, FULL_MASK, True,
                     captured_calls=counts[(bs, T)])
        )

    # Regression-only rows (production:false) — synthetic edges + fallback domain.
    # (min/max captured T are already production rows above; the off-shape guard
    #  conditions — draft!=2, non-contiguous parent_list, wrong selected_index
    #  shape — are exercised directly in bench/correctness.py, which builds those
    #  draft-specific tensors without complicating make_case.)
    rows.append(base_row("edge_bs1_seq0", 1, 4, FULL_MASK, False))            # seq_len=0 -> T=4
    rows.append(base_row("edge_bs10_seq0", 10, 40, FULL_MASK, False))         # all seq_len=0, bs=10
    rows.append(base_row("fallback_qlen_only_bs4", 4, 16, {**FULL_MASK, "tree_mask_mode": 1},
                         False, fallback_expected=True))                       # mode!=FULL_MASK -> baseline
    rows.append(base_row("fallback_noncontig_vsl_bs4", 4, 64, FULL_MASK, False,
                         noncontiguous="verified_seq_len", fallback_expected=True))

    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    prod = sum(1 for r in rows if r["production"])
    print(f"wrote {OUT}: {len(rows)} rows ({prod} production, {len(rows)-prod} regression)")


if __name__ == "__main__":
    main()
