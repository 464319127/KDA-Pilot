#!/usr/bin/env python3
"""Standalone correctness suite for fp8_scaled_mm (run BEFORE any benchmark).

For every workloads.json row: build synthetic inputs (seeded), run the recovered
baseline and the candidate through the shared ABI into POISONED output buffers,
and check the candidate against (a) an independent fp32 oracle and (b) the
baseline, plus shape / dtype / contiguity / NaN-Inf, and the dispatch route.

Oracle (per llm_correctness_contract.md FP8 GEMM):
    out[m,n] = ( sum_k A_fp8[m,k] * B_fp8[k,n] ) * scale_a[m] * scale_b[n]
computed by dequantizing A,B to fp32, matmul in fp32, applying scales, casting
to the output dtype. Tolerance: bf16 atol=0.07 rtol=0.02 (loose enough to absorb
fp8 tensor-core accumulation-order differences across K).

Route invariant: when route==0 (baseline fallback) the candidate must be
BIT-IDENTICAL to the baseline (it called the same impl); when route==1
(specialized fast path) it must match the oracle and the baseline within tol.

Exit 0 iff all rows pass. Run on the pinned remote B200 GPU (REMOTE_GPU_ID=3).
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import adapter  # noqa: E402
import build_ext  # noqa: E402

WORKLOADS = _HERE / "workloads.json"
_DT = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def oracle(inp, out_dtype):
    acc = inp["a"].float() @ inp["b"].float()            # [M,N] fp32
    acc = acc * inp["scale_a"].float() * inp["scale_b"].float().t()
    return acc.to(out_dtype)


def check_row(wl, device) -> tuple[bool, str]:
    out_dtype = _DT[wl.get("scalars", {}).get("out_dtype", "bfloat16")]
    contig_b = wl.get("strides", {}).get("b") == "row_major"
    has_bias = wl.get("scalars", {}).get("bias") is not None
    atol, rtol = wl.get("atol", 0.07), wl.get("rtol", 0.02)

    inp = adapter.make_inputs(wl, device=device, seed=wl.get("seed", 0))
    route = build_ext.route(inp["a"], inp["b"], inp["scale_a"], inp["scale_b"],
                            torch.empty((inp["M"], inp["N"]), device=device, dtype=out_dtype))

    # Contract-boundary edge rows (the recovered ABI rejects / does not carry these):
    # the candidate must NOT claim a fast path for them.
    if contig_b:
        if route != 0:
            return False, "contiguous-B must route to fallback (route!=0)"
        try:
            o = torch.empty((inp["M"], inp["N"]), device=device, dtype=out_dtype)
            build_ext.baseline(inp["a"], inp["b"], inp["scale_a"], inp["scale_b"], o)
        except Exception:
            return True, "contiguous-B rejected by contract as expected"
        return False, "contiguous-B should have been rejected (column-major required)"
    if has_bias:
        return (route == 0), ("bias!=None routes to fallback" if route == 0
                              else "bias!=None must route to fallback (route!=0)")

    # Valid input: run baseline + candidate into poisoned buffers.
    out_b = torch.full((inp["M"], inp["N"]), float("nan"), device=device, dtype=out_dtype)
    out_c = torch.full((inp["M"], inp["N"]), float("nan"), device=device, dtype=out_dtype)
    build_ext.baseline(inp["a"], inp["b"], inp["scale_a"], inp["scale_b"], out_b)
    build_ext.candidate(inp["a"], inp["b"], inp["scale_a"], inp["scale_b"], out_c)
    torch.cuda.synchronize()

    ref = oracle(inp, out_dtype).float()
    bf, cf = out_b.float(), out_c.float()
    # structural
    if out_c.shape != (inp["M"], inp["N"]) or out_c.dtype != out_dtype:
        return False, f"candidate shape/dtype wrong: {tuple(out_c.shape)}/{out_c.dtype}"
    if not out_c.is_contiguous():
        return False, "candidate output not contiguous (row-major expected)"
    if torch.isnan(cf).any() or torch.isinf(cf).any():
        return False, "candidate NaN/Inf (poison not fully overwritten)"
    if torch.isnan(bf).any():
        return False, "baseline NaN/Inf (poison not fully overwritten)"

    def within(x):
        return bool(((x - ref).abs() <= atol + rtol * ref.abs()).all())

    if not within(bf):
        return False, f"baseline vs oracle exceeds tol (max {float((bf-ref).abs().max()):.4f})"
    if not within(cf):
        return False, f"candidate vs oracle exceeds tol (max {float((cf-ref).abs().max()):.4f})"
    if route == 0:
        if not torch.equal(out_b, out_c):
            return False, "route==0 (fallback) but candidate != baseline bit-for-bit"
    else:
        if not bool(((bf - cf).abs() <= atol + rtol * bf.abs()).all()):
            return False, "route==1 candidate vs baseline exceeds tol"
    return True, f"ok (route={route})"


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    rows = json.loads(WORKLOADS.read_text())
    build_ext.get_ext()  # build once before timing any row
    npass = nfail = 0
    fails = []
    for wl in rows:
        try:
            ok, msg = check_row(wl, device)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"EXC {type(e).__name__}: {e}"
        tag = "PASS" if ok else "FAIL"
        if ok:
            npass += 1
        else:
            nfail += 1
            fails.append((wl["id"], msg))
        print(f"[{tag}] {wl['id']:<24} {wl.get('regime','?'):<12} {msg}")
    print(f"\n{npass} passed, {nfail} failed (of {len(rows)})")
    if fails:
        print("FAILURES:")
        for i, m in fails:
            print(f"  {i}: {m}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
