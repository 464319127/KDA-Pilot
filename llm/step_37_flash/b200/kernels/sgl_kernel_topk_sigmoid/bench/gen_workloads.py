"""Freeze bench/workloads.json from the captured evidence.

Deduplicates the 248 captured variants (docs/evidence.json) into distinct
(num_tokens, num_experts, topk, dtype, contiguity, scalar) rows. Every captured variant for
this kernel has the same structure — gating [N,288] fp32 contiguous, topk_weights [N,8] fp32,
topk_indices [N,8] i32, correction_bias [288] fp32, renormalize=True — and differs only in N
(and the cuda device, which we collapse since the benchmark pins one idle GPU). So the
production rows are one per distinct N, carrying the aggregate captured call count.

Regression rows (production=false, excluded from the headline geomean) exercise the baseline
fallback path and route=0: non-contiguous gating, bf16/fp16 gating, a power-of-two expert
count (different baseline path), topk!=8, renormalize=False, and missing bias.

Run locally (no GPU needed): python3 bench/gen_workloads.py
"""

from __future__ import annotations

import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
_EVIDENCE = _ROOT / "docs" / "evidence.json"
_OUT = _HERE / "workloads.json"

NUM_EXPERTS = 288
TOPK = 8
FP32_ATOL = 1e-5
FP32_RTOL = 1e-5


def captured_token_counts() -> dict[int, int]:
    """Return {num_tokens: aggregate_call_count} over the captured variants."""
    data = json.loads(_EVIDENCE.read_text())
    counts: dict[int, int] = {}
    for v in data.get("variants", []):
        n = None
        for t in v.get("args", [{}])[0].get("tensors", []):
            if t.get("name") == "arg[0]":
                n = int(t["shape"][0])
                break
        if n is None:
            continue
        counts[n] = counts.get(n, 0) + int(v.get("call_count", 0))
    return dict(sorted(counts.items()))


def regime(n: int) -> str:
    if n == 1:
        return "decode"
    if n <= 80:
        return "mid"
    return "large_prefill"


def build_rows() -> list[dict]:
    rows: list[dict] = []
    counts = captured_token_counts()
    for i, (n, calls) in enumerate(counts.items()):
        rows.append({
            "id": f"prod_n{n}",
            "production": True,
            "regime": regime(n),
            "function": "topk_sigmoid",
            "num_tokens": n,
            "num_experts": NUM_EXPERTS,
            "topk": TOPK,
            "dtype": "float32",
            "renormalize": True,
            "has_bias": True,
            "contiguous": True,
            "captured_call_count": calls,
            "atol": FP32_ATOL,
            "rtol": FP32_RTOL,
            "seed": 1000 + i,
        })

    # Regression / fallback rows (production=false; not in the headline geomean). These MUST
    # route to the baseline (route==0) and stay correct — proving the safe fallback.
    fallbacks = [
        {"id": "fb_noncontig", "note": "non-contiguous gating (strided view)", "contiguous": False},
        {"id": "fb_bf16", "note": "bf16 gating dtype", "dtype": "bfloat16"},
        {"id": "fb_fp16", "note": "fp16 gating dtype", "dtype": "float16"},
        {"id": "fb_experts64", "note": "power-of-two expert count (different baseline path)", "num_experts": 64},
        {"id": "fb_topk4", "note": "topk!=8", "topk": 4},
        {"id": "fb_no_renorm", "note": "renormalize=False", "renormalize": False},
    ]
    # Note: the captured contract always supplies correction_bias, and the local TVM-FFI ABI
    # passes it as a required TensorView, so a "missing bias" (None) case is out of the ABI's
    # domain and is intentionally not generated here.
    for j, fb in enumerate(fallbacks):
        row = {
            "id": fb["id"],
            "production": False,
            "regime": "fallback",
            "function": "topk_sigmoid",
            "num_tokens": 64,
            "num_experts": NUM_EXPERTS,
            "topk": TOPK,
            "dtype": "float32",
            "renormalize": True,
            "has_bias": True,
            "contiguous": True,
            "note": fb["note"],
            "atol": FP32_ATOL,
            "rtol": FP32_RTOL,
            "seed": 2000 + j,
        }
        for k, val in fb.items():
            if k not in ("id", "note"):
                row[k] = val
        rows.append(row)
    return rows


def main() -> None:
    rows = build_rows()
    _OUT.write_text(json.dumps(rows, indent=2) + "\n")
    prod = sum(1 for r in rows if r["production"])
    print(f"wrote {_OUT} : {len(rows)} rows ({prod} production, {len(rows) - prod} regression)")
    print("production num_tokens:", [r["num_tokens"] for r in rows if r["production"]])


if __name__ == "__main__":
    main()
