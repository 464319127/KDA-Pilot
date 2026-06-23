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
OUT = os.path.join(HERE, "workloads.json")

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

    rows = []
    # Production rows: every distinct captured (M, K, N), sorted by calls desc.
    for (m, k, n), c in sorted(calls.items(), key=lambda kv: -kv[1]):
        headline = (m in HEADLINE_M) and ((k, n) in HEADLINE_KN)
        rows.append({
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

    # Regression edge rows (production: false) from llm_correctness_contract.md.
    # These exercise the fallback path; the candidate must route them to baseline.
    # Representative (K, N) reused from the captured set so the baseline supports them.
    edge_specs = [
        ("edge_bias_m32_k1024_n8192", 32, 1024, 8192, {"bias": "bfloat16"}, "bias != None must fall back / be handled"),
        ("edge_contigB_m1_k1024_n8192", 1, 1024, 8192, {"b_contiguous": True}, "contiguous B (not column-major) must fall back"),
        ("edge_outhalf_m1_k1024_n8192", 1, 1024, 8192, {"out_dtype": "float16"}, "non-bf16 out_dtype must fall back / be handled"),
        ("edge_m2_k8192_n512", 2, 8192, 512, {}, "small even M not in captured set (route robustness)"),
        ("edge_m512_k512_n2048", 512, 512, 2048, {}, "rare (K,N)=512x2048 at a boundary M"),
    ]
    for rid, m, k, n, extra_scalars, why in edge_specs:
        b_contig = extra_scalars.get("b_contiguous", False)
        out_dt = extra_scalars.get("out_dtype", "bfloat16")
        bias = extra_scalars.get("bias", None)
        rows.append({
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

    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")

    prod = [r for r in rows if r["production"]]
    head = [r for r in prod if r["headline"]]
    total_calls = sum(r["calls"] for r in prod)
    by_regime = defaultdict(int)
    for r in prod:
        by_regime[r["regime"]] += 1
    print(f"wrote {OUT}")
    print(f"  production rows : {len(prod)} (covering {total_calls} captured calls)")
    print(f"  headline rows   : {len(head)}")
    print(f"  edge rows       : {len(rows) - len(prod)}")
    print(f"  by regime       : {dict(by_regime)}")
    head_calls = sum(r['calls'] for r in head)
    print(f"  headline covers : {head_calls} calls ({100*head_calls/total_calls:.1f}% of production)")


if __name__ == "__main__":
    main()
