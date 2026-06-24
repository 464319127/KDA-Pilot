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


def oracle(inp, out_dtype, bias=None):
    acc = inp["a"].float() @ inp["b"].float()            # [M,N] fp32
    acc = acc * inp["scale_a"].float() * inp["scale_b"].float().t()
    if bias is not None:
        acc = acc + bias.float().reshape(1, -1)
    return acc.to(out_dtype)


def expected_route(wl) -> int:
    """The intended dispatch route for a workload, mirroring the candidate's
    covers_m1_gemv predicate. Asserted in check_row so a silent fallback can never
    masquerade as a promoted route (AC-5)."""
    s = wl["shapes"]
    M, K, N = s["M"], s["K"], s["N"]
    contig_b = wl.get("strides", {}).get("b") == "row_major"
    out_bf16 = wl.get("scalars", {}).get("out_dtype", "bfloat16") == "bfloat16"
    if contig_b or not out_bf16 or K % 16 != 0:
        return 0  # contract violation -> baseline fallback
    if M == 1:
        return 1 if not (K >= 4096 and N >= 3072) else 0  # M=1 GEMV (route 1)
    # Small-M swap-AB (route 2) is disabled (Round 2 evidence-backed no-go);
    # small-M falls back to baseline (route 0).
    return 0


def check_row(wl, device) -> tuple[bool, str]:
    out_dtype = _DT[wl.get("scalars", {}).get("out_dtype", "bfloat16")]
    contig_b = wl.get("strides", {}).get("b") == "row_major"
    atol, rtol = wl.get("atol", 0.07), wl.get("rtol", 0.02)

    inp = adapter.make_inputs(wl, device=device, seed=wl.get("seed", 0))
    route = build_ext.route(inp["a"], inp["b"], inp["scale_a"], inp["scale_b"],
                            torch.empty((inp["M"], inp["N"]), device=device, dtype=out_dtype))

    # AC-5: the actual route must match the intended dispatch table for EVERY row,
    # so a covered M=1 winner can never silently degrade to the baseline (or vice versa).
    exp = expected_route(wl)
    if route != exp:
        return False, f"route {route} != expected {exp} (dispatch-table mismatch)"

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


def negative_route_cases(device) -> list:
    """Malformed inputs that MUST route to fallback (route==0) AND that the
    baseline rejects (no silent wrong compute). Exercises the hardened dispatch
    predicate (AC-3/AC-5). Each `neg_dtype_*`/`neg_scale_*` case would have
    misrouted (route==1) under the round-0 8-bit-only / size(0)-only check."""
    M, K, N = 1, 1024, 8192

    def f8(shape, dt=torch.float8_e4m3fn, dev=device):
        return torch.randn(shape, device=dev, dtype=torch.float32).to(dt)

    a = f8((M, K))
    b = f8((N, K)).t()  # [K,N] column-major
    sa = torch.rand((M, 1), device=device, dtype=torch.float32)
    sb = torch.rand((N, 1), device=device, dtype=torch.float32)
    out = torch.empty((M, N), device=device, dtype=torch.bfloat16)

    results = []

    def expect_fallback(name, aa, bb, saa, sbb, oo, expect_reject):
        r = build_ext.route(aa, bb, saa, sbb, oo)
        if r != 0:
            results.append((name, False, f"route=={r}, must be 0 (fallback)"))
            return
        if expect_reject:
            try:
                build_ext.candidate(aa, bb, saa, sbb, oo)
                torch.cuda.synchronize()
                results.append((name, False, "baseline did not reject malformed input"))
                return
            except Exception:
                pass  # baseline correctly rejects -> no silent wrong compute
        results.append((name, True, "route=0" + (" + baseline-rejected" if expect_reject else "")))

    expect_fallback("neg_dtype_e5m2_A", f8((M, K), torch.float8_e5m2), b, sa, sb, out, True)
    expect_fallback("neg_dtype_uint8_A", torch.zeros((M, K), dtype=torch.uint8, device=device), b, sa, sb, out, True)
    expect_fallback("neg_scale_a_rank_M2", a, b, torch.rand((M, 2), device=device, dtype=torch.float32), sb, out, True)
    expect_fallback("neg_scale_b_rank_N2", a, b, sa, torch.rand((N, 2), device=device, dtype=torch.float32), out, True)
    expect_fallback("neg_dtype_e5m2_B", a, f8((N, K), torch.float8_e5m2).t(), sa, sb, out, True)
    expect_fallback("neg_out_fp16", a, b, sa, sb, torch.empty((M, N), device=device, dtype=torch.float16), False)
    # Padded/sliced column-major B: stride(0)==1 but stride(1)=K+16 != K. The GEMV
    # hardcodes leading dim K, so this must fall back (route 0), not silently read
    # wrong columns. (Bphys [N,K+16] contiguous, sliced to [N,K], transposed.)
    b_pad = f8((N, K + 16))[:, :K].t()
    expect_fallback("neg_padded_B", a, b_pad, sa, sb, out, False)
    # CPU input must be rejected BEFORE any forced CUDA view (calls candidate, not just route).
    a_cpu = torch.randn((M, K), dtype=torch.float32).to(torch.float8_e4m3fn)  # on CPU
    expect_fallback("neg_cpu_input_A", a_cpu, b, sa, sb, out, True)
    # mixed-device: scale_b on a second GPU must be rejected by the device guard (calls candidate).
    if torch.cuda.device_count() >= 2:
        expect_fallback("neg_mixed_device", a, b, sa, sb.to("cuda:1"), out, True)
    else:
        results.append(("neg_mixed_device", True, "skipped (single visible GPU); device guard enforced in require_fp8_contract"))
    return results


def bias_edge_test(device):
    """AC-3.1 bias edge, exercised through the CANDIDATE fallback route. Uses an
    otherwise-covered M=1 winner (M=1,K=1024,N=8192): a biased call must NOT take
    the bias-unaware M=1 GEMV fast path — route_bias==0 — and must route to the
    recovered baseline with bias and be numerically correct
    (out=(A@B)*scale_a*scale_b+bias). Verifies candidate_bias vs the fp32 oracle,
    baseline_bias vs the oracle, and candidate==baseline."""
    M, K, N = 1, 1024, 8192
    wl = {"shapes": {"M": M, "K": K, "N": N}, "strides": {"b": "column_major"},
          "scalars": {"out_dtype": "bfloat16"}, "seed": 0}
    inp = adapter.make_inputs(wl, device=device, seed=0)
    bias = torch.randn((N,), device=device, dtype=torch.bfloat16)
    out_b = torch.full((M, N), float("nan"), device=device, dtype=torch.bfloat16)
    out_c = torch.full((M, N), float("nan"), device=device, dtype=torch.bfloat16)
    # A biased call must route to baseline (the M=1 GEMV/swap-AB fast paths are bias-unaware).
    r = build_ext.route_bias(inp["a"], inp["b"], inp["scale_a"], inp["scale_b"], bias, out_c)
    build_ext.baseline_bias(inp["a"], inp["b"], inp["scale_a"], inp["scale_b"], bias, out_b)
    build_ext.candidate_bias(inp["a"], inp["b"], inp["scale_a"], inp["scale_b"], bias, out_c)
    torch.cuda.synchronize()
    ref = oracle(inp, torch.bfloat16, bias=bias).float()
    bf, cf = out_b.float(), out_c.float()
    if torch.isnan(cf).any() or torch.isinf(cf).any():
        return ("bias_edge", False, "candidate_bias NaN/Inf (poison not overwritten)")
    tol = lambda x, y: bool(((x - y).abs() <= 0.07 + 0.02 * y.abs()).all())
    base_ok, cand_ok, eq_ok = tol(bf, ref), tol(cf, ref), tol(cf, bf)
    ok = (r == 0) and base_ok and cand_ok and eq_ok
    return ("bias_edge", ok,
            f"route_bias={r}; baseline_vs_oracle={base_ok}; candidate_bias_vs_oracle={cand_ok}; candidate==baseline={eq_ok}")


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
    # Negative-route (malformed-input) tests for the hardened dispatch predicate,
    # plus the AC-3.1 bias edge.
    n_rows = len(rows)
    extra = negative_route_cases(device) + [bias_edge_test(device)]
    for name, ok, msg in extra:
        tag = "PASS" if ok else "FAIL"
        if ok:
            npass += 1
        else:
            nfail += 1
            fails.append((name, msg))
        print(f"[{tag}] {name:<24} {'neg-route':<12} {msg}")
    print(f"\n{npass} passed, {nfail} failed (of {npass + nfail}; {n_rows} workload rows + negatives)")
    if fails:
        print("FAILURES:")
        for i, m in fails:
            print(f"  {i}: {m}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
