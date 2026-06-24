"""Standalone correctness suite for topk_sigmoid (run before any benchmark counts).

Checks, over the frozen production rows + regression rows (workloads.json) + constructed
tie/edge rows:

  1. candidate vs the recovered baseline  -> selected ids EXACT, weights fp32 atol/rtol 1e-5
     (the baseline is the authoritative oracle, esp. for tie-break).
  2. candidate vs an independent fp32 torch oracle (sigmoid+bias selection, unbiased weights,
     renormalize) on random production rows -> ids exact (no ties at random), weights tolerance.
  3. route coverage: production rows take the candidate fast path (route==1); regression rows
     fall back to the baseline (route==0) and stay correct. Proves no silent fallback.
  4. poison + in-place: output buffers are NaN/-17 poisoned before each run; after the candidate
     runs, weights are finite and indices are valid experts (full write, no stale entries).
  5. gating_output (read-only) is not mutated by either side.

Run on the target GPU: python3 bench/correctness.py   (exits non-zero on any failure)
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

import adapter
import build_ext

_HERE = pathlib.Path(__file__).resolve().parent
_WORKLOADS = _HERE / "workloads.json"
NUM_EXPERTS = 288
TOPK = 8


def torch_oracle(gating_f32: torch.Tensor, bias: torch.Tensor, topk: int, renormalize: bool):
    """Reference (matches upstream test_topk_sigmoid_renormalize_correction_bias)."""
    sig = torch.sigmoid(gating_f32.float())
    scores = sig + bias.float().unsqueeze(0)
    _, idx = torch.topk(scores, topk, dim=-1)
    w = sig.gather(1, idx)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return w.float(), idx.to(torch.int32)


def _poison(weights: torch.Tensor, indices: torch.Tensor) -> None:
    weights.fill_(float("nan"))
    indices.fill_(-17)


def _check_pair(name, w_b, idx_b, w_c, idx_c, atol, rtol, errs):
    if not torch.equal(idx_b.to(torch.int64), idx_c.to(torch.int64)):
        ndiff = int((idx_b.to(torch.int64) != idx_c.to(torch.int64)).sum().item())
        errs.append(f"[{name}] index mismatch: {ndiff} entries differ (exact required)")
        return False
    if not torch.allclose(w_c.float(), w_b.float(), atol=atol, rtol=rtol):
        mx = float((w_b.float() - w_c.float()).abs().max().item())
        errs.append(f"[{name}] weight mismatch max_abs={mx:.3e} (atol={atol},rtol={rtol})")
        return False
    return True


def run_row(workload, device, errs) -> bool:
    seed = 12345 + int(workload.get("seed", 0))
    case = adapter.make_case(workload, device=device, seed=seed)
    gating, renorm, bias = case.inputs
    gating_clone = gating.clone()
    name = workload["id"]
    ok = True

    # route coverage: production -> fast path (1); regression -> fallback (0).
    r = build_ext.route(case.baseline_outputs[0], case.baseline_outputs[1], gating, renorm, bias)
    expect_route = 1 if workload.get("production", True) else 0
    if r != expect_route:
        errs.append(f"[{name}] route={r}, expected {expect_route} (fast-path coverage / fallback proof)")
        ok = False

    # baseline + candidate on poisoned, separate output buffers.
    _poison(*case.baseline_outputs)
    _poison(*case.candidate_outputs)
    adapter.call_baseline(workload, case.inputs, case.baseline_outputs)
    adapter.call_candidate(workload, case.inputs, case.candidate_outputs)
    torch.cuda.synchronize()

    w_b, idx_b = case.baseline_outputs
    w_c, idx_c = case.candidate_outputs

    if not _check_pair(name + ":cand-vs-base", w_b, idx_b, w_c, idx_c,
                       workload.get("atol", 1e-5), workload.get("rtol", 1e-5), errs):
        ok = False

    # poison/in-place: candidate fully wrote outputs (no NaN weights, valid expert ids).
    if not torch.isfinite(w_c).all().item():
        errs.append(f"[{name}] candidate left NaN/Inf in weights (partial write?)"); ok = False
    if (idx_c < 0).any().item() or (idx_c >= workload["num_experts"]).any().item():
        errs.append(f"[{name}] candidate left invalid expert ids (poison -17 leftover?)"); ok = False

    # read-only gating must be unchanged by both sides.
    if not torch.equal(gating, gating_clone):
        errs.append(f"[{name}] gating_output was mutated (must be read-only)"); ok = False

    # independent oracle on random fp32 production rows (no ties at random -> ids exact).
    if workload.get("production", True) and workload.get("dtype", "float32") == "float32":
        w_o, idx_o = torch_oracle(gating, bias, workload["topk"], bool(workload.get("renormalize", True)))
        if not _check_pair(name + ":cand-vs-oracle", w_o, idx_o, w_c, idx_c, 1e-4, 1e-4, errs):
            ok = False
    return ok


def nobias_row(device, errs) -> bool:
    """Missing-bias fallback: with correction_bias=None the candidate must route to the
    baseline (route_nobias==0) and stay correct (candidate==baseline exact + vs a no-bias oracle)."""
    n = 64
    gen = torch.Generator(device=device).manual_seed(4242)
    g = torch.randn((n, NUM_EXPERTS), dtype=torch.float32, device=device, generator=gen)
    wb = torch.empty((n, TOPK), dtype=torch.float32, device=device)
    ib = torch.empty((n, TOPK), dtype=torch.int32, device=device)
    wc = torch.empty_like(wb)
    ic = torch.empty_like(ib)
    _poison(wb, ib)
    _poison(wc, ic)
    ok = True

    r = build_ext.route_nobias(wb, ib, g, 1)
    if r != 0:
        errs.append(f"[fb_missing_bias] route_nobias={r}, expected 0 (no fast path without bias)")
        ok = False

    build_ext.baseline_nobias(wb, ib, g, 1)
    build_ext.candidate_nobias(wc, ic, g, 1)
    torch.cuda.synchronize()
    if not _check_pair("fb_missing_bias:cand-vs-base", wb, ib, wc, ic, 1e-5, 1e-5, errs):
        ok = False
    if not torch.isfinite(wc).all().item():
        errs.append("[fb_missing_bias] candidate left NaN/Inf in weights"); ok = False
    if (ic < 0).any().item() or (ic >= NUM_EXPERTS).any().item():
        errs.append("[fb_missing_bias] candidate left invalid expert ids"); ok = False

    # Independent no-bias oracle: sigmoid top-k (no bias add/subtract), renormalized.
    sig = torch.sigmoid(g.float())
    w_o, idx_o = torch.topk(sig, TOPK, dim=-1)
    w_o = w_o / w_o.sum(dim=-1, keepdim=True)
    if not _check_pair("fb_missing_bias:cand-vs-oracle", w_o.float(), idx_o.to(torch.int32), wc, ic, 1e-4, 1e-4, errs):
        ok = False
    return ok


def fp16_bias_fallback_row(device, errs) -> bool:
    """Fallback-safety: an fp16 correction_bias is off-domain (the fast path requires fp32 bias), so it
    must route to the baseline (route==0). The local ABI bridge must preserve the real fp16 dtype so the
    vendored baseline's `correction_bias must be float32` TORCH_CHECK fires cleanly on BOTH sides — not
    re-tag the fp16 storage as fp32 and read past the allocation. Run last: a regression here raises
    host-side before any launch, but guarding placement keeps a buggy build from affecting other rows."""
    n = 16
    gen = torch.Generator(device=device).manual_seed(8765)
    g = torch.randn((n, NUM_EXPERTS), dtype=torch.float32, device=device, generator=gen)
    bias16 = torch.randn((NUM_EXPERTS,), dtype=torch.float16, device=device, generator=gen)
    w = torch.empty((n, TOPK), dtype=torch.float32, device=device)
    idx = torch.empty((n, TOPK), dtype=torch.int32, device=device)
    ok = True

    if build_ext.route(w, idx, g, 1, bias16) != 0:
        errs.append("[fp16_bias_fallback] route != 0 (fp16 bias must be off the fast path)"); ok = False

    def raises_cleanly(fn) -> bool:
        try:
            fn()
            torch.cuda.synchronize()
            return False
        except RuntimeError:
            return True

    if not raises_cleanly(lambda: build_ext.baseline(w, idx, g, 1, bias16)):
        errs.append("[fp16_bias_fallback] baseline did not reject fp16 bias (expected dtype TORCH_CHECK)"); ok = False
    if not raises_cleanly(lambda: build_ext.candidate(w, idx, g, 1, bias16)):
        errs.append("[fp16_bias_fallback] candidate did not reject fp16 bias cleanly (possible OOB / wrong-element-size reinterpretation)"); ok = False
    return ok


def tie_rows(device):
    """Constructed tie/edge rows; validated against the authoritative baseline (torch.topk
    tie-break is unstable, so the baseline — not the oracle — is the reference here)."""
    E, N = NUM_EXPERTS, 4
    rows = []
    # 1) all-equal logits + zero bias -> all scores equal -> lowest-index experts win.
    g = torch.zeros((N, E), dtype=torch.float32, device=device)
    b = torch.zeros((E,), dtype=torch.float32, device=device)
    rows.append(("tie_identical", g, b))
    # 2) all-equal logits + block-tied biases -> score ties broken by index.
    b2 = torch.zeros((E,), dtype=torch.float32, device=device)
    b2[:16] = 1.0  # first 16 experts tie at the top -> expect ids 0..7
    rows.append(("tie_bias_blocks", g.clone(), b2))
    # 3) equal sigmoid+bias but different raw sigmoid (weights must be UNBIASED, not score).
    g3 = torch.zeros((N, E), dtype=torch.float32, device=device)
    g3[:, 0] = 2.0          # sigmoid(2)=~0.8808
    g3[:, 1] = -2.0         # sigmoid(-2)=~0.1192
    b3 = torch.zeros((E,), dtype=torch.float32, device=device)
    b3[0] = 0.0
    b3[1] = 0.8808 - 0.1192  # make score[1] ~= score[0]; selection ties, weights differ
    rows.append(("tie_equal_score_diff_sigmoid", g3, b3))
    # 4) near-ties from tiny perturbations.
    g4 = torch.randn((N, E), dtype=torch.float32, device=device, generator=torch.Generator(device=device).manual_seed(7))
    g4[:, 5] = g4[:, 6]  # force an exact sigmoid tie between experts 5 and 6
    b4 = torch.zeros((E,), dtype=torch.float32, device=device)
    rows.append(("tie_near", g4, b4))
    return rows


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    workloads = json.loads(_WORKLOADS.read_text())
    errs: list[str] = []
    npass = 0
    ntotal = 0

    for w in workloads:
        ntotal += 1
        if run_row(w, device, errs):
            npass += 1

    # missing-bias fallback row
    ntotal += 1
    if nobias_row(device, errs):
        npass += 1

    for name, g, b in tie_rows(device):
        ntotal += 1
        w_b = torch.empty((g.shape[0], TOPK), dtype=torch.float32, device=device)
        idx_b = torch.empty((g.shape[0], TOPK), dtype=torch.int32, device=device)
        w_c = torch.empty_like(w_b)
        idx_c = torch.empty_like(idx_b)
        _poison(w_b, idx_b); _poison(w_c, idx_c)
        build_ext.baseline(w_b, idx_b, g, 1, b)
        build_ext.candidate(w_c, idx_c, g, 1, b)
        torch.cuda.synchronize()
        if _check_pair(name, w_b, idx_b, w_c, idx_c, 1e-5, 1e-5, errs):
            npass += 1

    # fp16-bias fallback safety (run last; see the function docstring)
    ntotal += 1
    if fp16_bias_fallback_row(device, errs):
        npass += 1

    print(f"correctness: {npass}/{ntotal} rows passed")
    if errs:
        print("FAILURES:")
        for e in errs:
            print("  -", e)
        return 1
    print("ALL CORRECTNESS CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
