#!/usr/bin/env python3
"""Freeze bench/workloads.json from docs/evidence.json for fast_topk_transform_fused.

Run: `python3 bench/gen_workloads.py` (deterministic; re-emits the frozen file).

Faithful to the recovered C++ contract (baseline/.../elementwise/topk.cu):
  fast_topk_transform_fused(score[B,N] f32, lengths[B] i32, page_table_size_1[S,M] i32,
                            cu_seqlens_q[S+1] i32, topk=2048, row_starts=None) -> dst[B,2048] i32
  - lengths is shape B (per token row); TORCH_CHECK(lengths.size(0)==score.size(0)).
  - src_page_table (page_table_size_1) is shape (S, M); S = cu_seqlens_q.size(0)-1.
  - is_decode = (row_starts is None and S == B); otherwise prefill (cu_seqlens maps token->seq).
  - naive path (length<=TopK): dst[i] = (i<length) ? src_page_table[i] : -1, so length must be
    <= M (page-table width) AND <= N (score width). We cap synthetic length to min(N, M).
  - score must have stride(1)==1; "non-contiguous" captures are row-strided (stride(0) > N).
  - ONE output tensor (dst_page_table); the capture's two identical (B,2048) records are logger
    duplication of the destination-passing buffer (confirmed by C++ + wrapper + DSA caller; see
    docs/baseline_source.md). Final confirmation = remote differential probe.

Dedup key = (B, N, score_contiguous, M, S) — the full distinguishing tuple (score dtype, scalar
kwargs, output count are constant across the capture). Call counts are summed; labels collected.
Every distinct captured shape is kept as a production row (no high-call shape dropped); top rows by
call volume are tagged headline. A synthetic regression grid (production:false) covers the edges.
"""
import json
import collections
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
KERNEL_DIR = BENCH_DIR.parent
EVIDENCE = KERNEL_DIR / "docs" / "evidence.json"
OUT = BENCH_DIR / "workloads.json"
TOPK = 2048
ROW_PAD = 8  # surrogate row padding for non-contiguous score: stride(0) = N + ROW_PAD


def score_stride(N, contiguous):
    # row-major; non-contiguous capture reproduced as a row slice of a wider buffer
    return [N, 1] if contiguous else [N + ROW_PAD, 1]


def base_row(rid, B, N, S, M, contiguous, *, production, headline, label,
             calls=None, score_dist="random", lengths_mode="full",
             page_table_mode="arange", row_starts_kind="none"):
    max_valid_length = min(N, M)  # cap so naive path never reads past src_page_table / score
    row = {
        "id": rid,
        "production": production,
        "headline": headline,
        "function": "fast_topk_transform_fused",
        "label": label,
        "shapes": {
            "score": [B, N],
            "lengths": [B],                      # per token row (C++: lengths.size(0)==B)
            "page_table_size_1": [S, M],         # per sequence (src_page_table rows == S)
            "cu_seqlens_q": [S + 1],
        },
        "dtypes": {"score": "float32", "lengths": "int32",
                   "page_table_size_1": "int32", "cu_seqlens_q": "int32", "out": "int32"},
        "strides": {
            "score": score_stride(N, contiguous),
            "score_contiguous": contiguous,
            "stride_source": "captured_contiguous" if contiguous else "surrogate_row_padded",
        },
        "scalars": {
            "topk": TOPK, "row_starts": None,
            "row_starts_kind": row_starts_kind,   # "none" | "tensor" (captured large-prefill rows)
            "page_table_mode": page_table_mode,   # "arange" | "permuted" (transform validation)
            "B": B, "N": N, "S": S, "M": M,
            "max_valid_length": max_valid_length,
            "is_decode": (S == B),
            "score_dist": score_dist, "lengths_mode": lengths_mode,
        },
        "outputs": [{"name": "dst_page_table", "shape": [B, TOPK], "dtype": "int32", "contiguous": True}],
        "atol": 0, "rtol": 0,
        "seed": 0,
    }
    if calls is not None:
        row["calls"] = calls
    return row


def main():
    d = json.loads(EVIDENCE.read_text())
    variants = d["variants"]

    groups = {}  # (B,N,contig,M,S) -> dict(calls, labels)
    for v in variants:
        arg = v["args"][0]
        tens = {t["name"]: t for t in arg["tensors"]}
        sc = tens.get("score")
        if sc is None:
            continue
        B, N = sc["shape"][0], sc["shape"][1]
        contig = bool(sc["is_contiguous"])
        cu = tens.get("cu_seqlens_q")
        S = (cu["shape"][0] - 1) if cu else B
        pt = tens.get("page_table_size_1")
        M = pt["shape"][1] if pt else N
        rs_kind = "tensor" if "row_starts" in tens else "none"  # captured row_starts scalar contract
        calls = int(v.get("call_count", 0))
        key = (B, N, contig, M, S, rs_kind)
        g = groups.setdefault(key, {"calls": 0, "labels": collections.Counter()})
        g["calls"] += calls
        g["labels"][v.get("label")] += calls

    rows = sorted(groups.items(), key=lambda kv: kv[1]["calls"], reverse=True)
    total_calls = sum(g["calls"] for _, g in rows)
    cum = 0
    workloads = []
    for (B, N, contig, M, S, rs_kind), g in rows:
        cum += g["calls"]
        headline = (cum <= 0.90 * total_calls) or (g["calls"] >= 100)
        tag = "cont" if contig else "strided"
        rs_tag = "_rs" if rs_kind == "tensor" else ""
        rid = f"ftt_B{B}_N{N}_M{M}_S{S}_{tag}{rs_tag}"
        wl = base_row(rid, B, N, S, M, contig, production=True, headline=bool(headline),
                      label=g["labels"].most_common(1)[0][0], calls=g["calls"], row_starts_kind=rs_kind)
        workloads.append(wl)
    n_prod = len(workloads)

    # ---- synthetic regression grid (production:false; correctness-only edges) ----
    reg = []

    def add(rid, B, N, S, M, contig, **kw):
        reg.append(base_row(f"reg_{rid}", B, N, S, M, contig,
                            production=False, headline=False, label="regression", **kw))

    add("N_lt_topk_tiny", 1, 1, 1, 1, True)                       # 1 candidate, decode
    add("N_lt_topk_two", 2, 2, 2, 2, True)
    add("N_eq_topk", 8, TOPK, 8, TOPK, True)                      # N==topk -> naive boundary
    add("N_gt_topk", 8, TOPK + 64, 8, TOPK + 64, True)            # N>topk -> radix path
    add("N_gt_topk_strided", 4, 5151, 4, 2088, False)             # large prefill-like, M<N, radix
    add("ties_small", 8, 256, 8, 256, True, score_dist="ties")    # equal-score tie-break (naive)
    add("ties_boundary", 8, TOPK + 128, 8, TOPK + 128, True, score_dist="ties")  # ties in radix
    add("len_zero", 2, 64, 2, 40, True, lengths_mode="zero")      # zero valid candidates
    add("len_one", 2, 64, 2, 40, True, lengths_mode="one")        # one valid candidate
    add("len_half", 8, 960, 8, 900, False, lengths_mode="half")   # lengths < M, strided
    add("prefill_small", 16, 448, 4, 409, False)                  # prefill S<B, strided
    add("prefill_decode_mix", 20, 704, 20, 700, True)             # decode S==B, contiguous
    add("permuted_naive", 8, 256, 8, 256, True, page_table_mode="permuted")       # non-linear transform (naive)
    add("permuted_radix", 8, 2112, 8, 2112, True, page_table_mode="permuted")     # non-linear transform (radix)
    add("row_starts_radix", 8, 2304, 8, 2176, False, row_starts_kind="tensor")    # ragged prefill, L=2176>topk
    workloads.extend(reg)

    OUT.write_text(json.dumps(workloads, indent=2) + "\n")
    head = [w for w in workloads if w["production"] and w["headline"]]
    print(f"wrote {OUT}")
    print(f"  total rows:      {len(workloads)}")
    print(f"  production rows: {n_prod} (dedup key (B,N,contig,M,S); covering {total_calls} calls)")
    print(f"  headline rows:   {len(head)}")
    print(f"  regression rows: {len(reg)}")
    # sanity: lengths never exceeds min(N,M); src_page_table never over-read
    bad = [w["id"] for w in workloads if w["scalars"]["max_valid_length"] > min(w["scalars"]["N"], w["scalars"]["M"])]
    assert not bad, bad
    want = [(8, 2112), (20, 2112), (8, 448), (20, 704), (2, 64), (2334, 2334), (3080, 3080), (2207, 5151)]
    present = {(w["scalars"]["B"], w["scalars"]["N"]) for w in workloads if w["production"]}
    print(f"  high-vol/large-prefill present: {[s for s in want if s in present]}")
    print(f"  MISSING (should be []): {[s for s in want if s not in present]}")


if __name__ == "__main__":
    main()
