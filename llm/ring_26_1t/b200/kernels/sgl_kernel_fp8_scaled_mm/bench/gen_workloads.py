#!/usr/bin/env python3
"""Freeze bench/workloads.json from the captured evidence.

Source of truth: ../docs/evidence.json (runtime SGLang interface capture of
sgl_kernel.fp8_scaled_mm for inclusionAI/Ring-2.6-1T on B200). Every distinct
(M, K, N) production shape is emitted as one deduplicated row. Device is
dropped (the run is pinned to one GPU); all variants are homogeneous on dtype
(fp8_e4m3 A/B, fp32 scales), contiguity (A row-major, B column-major, scales
contiguous), out_dtype (bf16) and bias (None), so those are constants here.

No tensor values are captured in evidence.json (only shape/dtype metadata), so
correctness/benchmark use documented synthetic inputs with a fixed seed.

This generator is deterministic; the emitted workloads.json is the frozen
artifact and must not be regenerated after tuning starts (see
llm/docs/standalone_llm_benchmark.md Workload Rules).

Regimes follow the baseline's own sm100 M-buckets (see docs/baseline_source.md):
  decode_tiny  : M <= 16        (baseline Gemm16,  CTA-M=64)
  decode_small : 16 < M <= 64   (baseline Gemm64,  CTA-M=64)
  medium       : 64 < M <= 256  (baseline Gemm256)
  prefill      : M > 256        (baseline GemmDefault)

Headline core (fast RLCR-iteration set, also production): the hot decode shapes
M in {1, 23, 32, 57} crossed with the five highest-call (K, N) dims.
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EVIDENCE = os.path.join(HERE, "..", "docs", "evidence.json")
OUT = os.path.join(HERE, "workloads.json")            # production-only (default benchmark input)
OUT_EDGES = os.path.join(HERE, "workloads_edges.json")  # correctness-only edge rows (the template benchmark cannot run the malformed ones)

ATOL, RTOL, SEED = 0.07, 0.02, 0  # bf16 tolerance per llm_correctness_contract.md
HEADLINE_M = {1, 23, 32, 57}
HEADLINE_KN = {(1024, 8192), (256, 8192), (8192, 512), (8192, 1024), (8192, 3072)}


def regime_of(m: int) -> str:
    if m <= 16:
        return "decode_tiny"
    if m <= 64:
        return "decode_small"
    if m <= 256:
        return "medium"
    return "prefill"


def main() -> None:
    with open(EVIDENCE) as f:
        ev = json.load(f)

    # Deduplicate captured variants into distinct (M, K, N), summing call counts.
    calls = defaultdict(int)
    for v in ev["variants"]:
        t = v["args"][0]["tensors"]
        m = t[0]["shape"][0]            # A = [M, K]
        k, n = t[1]["shape"][0], t[1]["shape"][1]  # B = [K, N]
        calls[(m, k, n)] += v.get("call_count", 0)

    prod_rows = []
    # Production rows: every distinct captured (M, K, N), sorted by calls desc.
    for (m, k, n), c in sorted(calls.items(), key=lambda kv: -kv[1]):
        headline = (m in HEADLINE_M) and ((k, n) in HEADLINE_KN)
        prod_rows.append({
            "id": f"m{m}_k{k}_n{n}",
            "production": True,
            "headline": headline,
            "function": "fp8_scaled_mm",
            "regime": regime_of(m),
            "shapes": {"M": m, "K": k, "N": n},
            "dtypes": {"a": "float8_e4m3fn", "b": "float8_e4m3fn",
                       "scale_a": "float32", "scale_b": "float32", "out": "bfloat16"},
            "strides": {"a": "row_major", "b": "column_major",
                        "scale_a": "contiguous", "scale_b": "contiguous", "out": "row_major"},
            "scalars": {"bias": None, "out_dtype": "bfloat16"},
            "calls": c,
            "atol": ATOL, "rtol": RTOL, "seed": SEED,
        })

    # Regression edge rows (production: false). These are shapes the adapter can
    # construct from standard inputs (varying stride / out dtype / M / (K,N)); the
    # candidate must route them to baseline. The malformed-input negatives that the
    # adapter cannot express as a normal case — wrong input dtype (fp8_e5m2/uint8),
    # malformed scale rank ([M,2]/[N,2]), and mixed-device — are exercised directly
    # in bench/correctness.py:negative_route_cases() (they assert route==0 and that
    # the baseline rejects them), since they require deliberately-malformed tensors.
    #
    # bias!=None is NOT an edge row: all 2720 captured variants are bias=None and
    # the recovered local ABI implements that captured contract (no bias channel),
    # so bias is outside the benchmarked interface (documented in docs/results.md).
    edge_specs = [
        ("edge_contigB_m1_k1024_n8192", 1, 1024, 8192, {"b_contiguous": True}, "contiguous B (not column-major) must be rejected by the contract"),
        ("edge_outhalf_m1_k1024_n8192", 1, 1024, 8192, {"out_dtype": "float16"}, "non-bf16 out_dtype must fall back (route 0)"),
        ("edge_m2_k8192_n512", 2, 8192, 512, {}, "small even M not in captured set (route robustness, fall back)"),
        ("edge_m512_k512_n2048", 512, 512, 2048, {}, "rare (K,N)=512x2048 at a boundary M (fall back)"),
    ]
    edge_rows = []
    for rid, m, k, n, extra_scalars, why in edge_specs:
        b_contig = extra_scalars.get("b_contiguous", False)
        out_dt = extra_scalars.get("out_dtype", "bfloat16")
        bias = extra_scalars.get("bias", None)
        edge_rows.append({
            "id": rid,
            "production": False,
            "headline": False,
            "function": "fp8_scaled_mm",
            "regime": "edge",
            "shapes": {"M": m, "K": k, "N": n},
            "dtypes": {"a": "float8_e4m3fn", "b": "float8_e4m3fn",
                       "scale_a": "float32", "scale_b": "float32", "out": out_dt},
            "strides": {"a": "row_major", "b": "row_major" if b_contig else "column_major",
                        "scale_a": "contiguous", "scale_b": "contiguous", "out": "row_major"},
            "scalars": {"bias": bias, "out_dtype": out_dt},
            "calls": 0,
            "atol": ATOL, "rtol": RTOL, "seed": SEED,
            "note": why,
        })

    # workloads.json holds ONLY the production rows so the unmodified template
    # `bench/benchmark.py` (which runs every row in its input) produces full-grid
    # results when run with the default input. The edge rows — including the
    # intentionally-malformed contiguous-B row that the baseline rejects — live in a
    # correctness-only file that bench/correctness.py reads in addition.
    with open(OUT, "w") as f:
        json.dump(prod_rows, f, indent=2)
        f.write("\n")
    with open(OUT_EDGES, "w") as f:
        json.dump(edge_rows, f, indent=2)
        f.write("\n")

    head = [r for r in prod_rows if r["headline"]]
    total_calls = sum(r["calls"] for r in prod_rows)
    by_regime = defaultdict(int)
    for r in prod_rows:
        by_regime[r["regime"]] += 1
    print(f"wrote {OUT} ({len(prod_rows)} production rows, covering {total_calls} captured calls)")
    print(f"wrote {OUT_EDGES} ({len(edge_rows)} correctness-only edge rows)")
    print(f"  headline rows   : {len(head)}")
    print(f"  by regime       : {dict(by_regime)}")
    head_calls = sum(r['calls'] for r in head)
    print(f"  headline covers : {head_calls} calls ({100*head_calls/total_calls:.1f}% of production)")


if __name__ == "__main__":
    main()
