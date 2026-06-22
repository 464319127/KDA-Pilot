"""Correctness grid for the grouped-top-k router kernel.

Validates, before any benchmark counts:
  * the recovered baseline against an independent PyTorch/bit-exact oracle, and
  * (when present) the native-CUDA candidate against the recovered baseline with
    EXACT-match ordered indices and weights within fp32 tolerance.

Coverage: every captured production shape (random, multiple seeds) + a
constructed value-edge grid (ties, equal sigmoid+bias via different bias,
negative bias, saturating/Inf logits, N=0, max N) + output poisoning, NaN/Inf,
shape/dtype/device checks, and an unsupported-parameter check (must raise like
the baseline).

Run:  CUDA_VISIBLE_DEVICES=0 python correctness.py
Exit 0 iff all checks pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from _jit_build import baseline_module, candidate_module, has_candidate

BENCH = Path(__file__).resolve().parent
DEV = "cuda"
ATOL, RTOL = 1e-5, 1e-5


# ─────────────────────────── independent oracle ───────────────────────────
def _pack_keys(biased_row: np.ndarray) -> np.ndarray:
    """Replicate the kernel's pack_val_idx ordering (monotonic float bits, and
    smaller index wins ties via 65535-idx) so the oracle's selection order is
    bit-exact with the kernel's iterative argmax."""
    vb = biased_row.astype(np.float32).view(np.uint32).astype(np.uint64)
    sign = (vb & np.uint64(0x80000000)) != 0
    vb = np.where(sign, vb ^ np.uint64(0xFFFFFFFF), vb ^ np.uint64(0x80000000))
    idx = np.arange(biased_row.shape[0], dtype=np.uint64)
    idxb = np.uint64(65535) - idx
    return (vb << np.uint64(32)) | idxb


def oracle(scores: torch.Tensor, bias: torch.Tensor, topk: int,
           renormalize: bool, scaling_factor: float):
    """Returns (weights[N,topk] f32, indices[N,topk] i32) in baseline selection order."""
    s = torch.sigmoid(scores.float())          # un-biased sigmoid weights
    biased = (s + bias.float()).cpu().numpy()
    s_np = s.cpu().numpy()
    N, E = biased.shape
    out_idx = np.empty((N, topk), dtype=np.int32)
    out_w = np.empty((N, topk), dtype=np.float32)
    for n in range(N):
        keys = _pack_keys(biased[n])
        order = np.argsort(keys)[::-1][:topk]   # descending packed key
        out_idx[n] = order.astype(np.int32)
        w = s_np[n, order].astype(np.float32)
        if renormalize:
            w = w / (w.sum(dtype=np.float32) + np.float32(1e-20))
        out_w[n] = (w * np.float32(scaling_factor)).astype(np.float32)
    return torch.from_numpy(out_w), torch.from_numpy(out_idx)


# ─────────────────────────── kernel invocation ───────────────────────────
def run_kernel(mod, scores, bias, topk, renormalize, scaling_factor,
               ng=1, tg=1, poison=True):
    N = scores.shape[0]
    tv = torch.empty((N, topk), dtype=torch.float32, device=scores.device)
    ti = torch.empty((N, topk), dtype=torch.int32, device=scores.device)
    if poison and N > 0:
        tv.fill_(float("nan"))
        ti.fill_(-17)
    mod.grouped_topk(scores, bias, tv, ti, ng, tg, topk, renormalize, scaling_factor)
    torch.cuda.synchronize()
    return tv, ti


class Checker:
    def __init__(self):
        self.fail = 0
        self.npass = 0

    def check(self, cond, msg):
        if cond:
            self.npass += 1
        else:
            self.fail += 1
            print(f"  FAIL: {msg}")

    def exact_idx(self, label, a_idx, b_idx):
        self.check(torch.equal(a_idx.cpu(), b_idx.cpu()),
                   f"{label}: ordered indices differ "
                   f"({int((a_idx.cpu()!=b_idx.cpu()).sum())} slots)")

    def weights_close(self, label, a_w, b_w, atol=ATOL, rtol=RTOL):
        a = a_w.float().cpu(); b = b_w.float().cpu()
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        self.check(ok, f"{label}: weights differ max_abs={float((a-b).abs().max()):.3e}")

    def no_poison(self, label, tv, ti):
        if tv.numel() == 0:
            return
        self.check(not torch.isnan(tv).any(), f"{label}: NaN survived in weights (unwritten slot)")
        self.check(not torch.isinf(tv).any(), f"{label}: Inf in weights")
        self.check(not (ti == -17).any(), f"{label}: poison sentinel -17 survived in indices")

    def shape_dtype(self, label, tv, ti, N, topk, dev):
        self.check(tuple(tv.shape) == (N, topk) and tv.dtype == torch.float32, f"{label}: weights shape/dtype")
        self.check(tuple(ti.shape) == (N, topk) and ti.dtype == torch.int32, f"{label}: indices shape/dtype")
        self.check(tv.device.type == dev and ti.device.type == dev, f"{label}: device")


def validate_case(ck: Checker, label, scores, bias, topk=8, renorm=True, scale=1.0,
                  cand=None):
    base = baseline_module()
    bv, bi = run_kernel(base, scores, bias, topk, renorm, scale)
    N = scores.shape[0]
    ck.shape_dtype(label + "[base]", bv, bi, N, topk, "cuda")
    ck.no_poison(label + "[base]", bv, bi)
    # baseline vs independent oracle (semantic cross-check)
    ov, oi = oracle(scores, bias, topk, renorm, scale)
    ck.exact_idx(label + "[base-vs-oracle]", bi, oi.to(scores.device))
    ck.weights_close(label + "[base-vs-oracle]", bv, ov.to(scores.device))
    # candidate vs baseline (the gating exact check)
    if cand is not None:
        cv, ci = run_kernel(cand, scores, bias, topk, renorm, scale)
        ck.shape_dtype(label + "[cand]", cv, ci, N, topk, "cuda")
        ck.no_poison(label + "[cand]", cv, ci)
        ck.exact_idx(label + "[cand-vs-base]", ci, bi)
        ck.weights_close(label + "[cand-vs-base]", cv, bv)


def main():
    torch.manual_seed(1234)
    ck = Checker()
    base = baseline_module()
    cand = candidate_module() if has_candidate() else None
    print(f"candidate present: {cand is not None}")

    # 1) captured production shapes (random, 2 seeds each)
    rows = [r for r in json.loads((BENCH / "workloads.json").read_text()) if r.get("production")]
    for r in rows:
        N, E, topk = r["shapes"]["num_tokens"], r["shapes"]["num_experts"], r["scalars"]["topk"]
        for seed in (0, 7):
            torch.manual_seed(seed * 100 + N)
            scores = torch.randn(N, E, dtype=torch.float32, device=DEV)
            bias = torch.randn(E, dtype=torch.float32, device=DEV)
            validate_case(ck, f"{r['id']}@s{seed}", scores, bias, topk, True, 1.0, cand)

    # 2) constructed value-edge grid
    E = 256
    bias0 = torch.zeros(E, dtype=torch.float32, device=DEV)
    # 2a) exact tie: two experts with identical score & zero bias -> smaller index wins
    sc = torch.full((1, E), -5.0, dtype=torch.float32, device=DEV)
    sc[0, 10] = 3.0; sc[0, 200] = 3.0  # tie between idx 10 and 200
    bv, bi = run_kernel(base, sc, bias0, 8, True, 1.0)
    ck.check(int(bi[0, 0]) == 10, f"tie: smaller index 10 picked first, got {int(bi[0,0])}")
    validate_case(ck, "tie_10_200", sc, bias0, 8, True, 1.0, cand)
    # 2b) equal sigmoid+bias via different bias values -> tie -> smaller index
    bias_eq = torch.zeros(E, dtype=torch.float32, device=DEV)
    sc2 = torch.full((1, E), -5.0, dtype=torch.float32, device=DEV)
    sc2[0, 5] = 1.0
    sc2[0, 100] = 2.0
    # make biased[5]==biased[100]: sigmoid(1)+b5 == sigmoid(2)+b100
    import math
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    bias_eq[5] = 0.0
    bias_eq[100] = sig(1.0) - sig(2.0)  # so sigmoid(2)+this == sigmoid(1)
    validate_case(ck, "equal_biased_via_bias", sc2, bias_eq, 8, True, 1.0, cand)
    # 2c) negative bias (biased scores go negative)
    torch.manual_seed(3)
    scn = torch.randn(4, E, dtype=torch.float32, device=DEV)
    biasn = torch.full((E,), -2.0, dtype=torch.float32, device=DEV)
    validate_case(ck, "negative_bias", scn, biasn, 8, True, 1.0, cand)
    # 2d) saturating + Inf logits
    scinf = torch.randn(2, E, dtype=torch.float32, device=DEV)
    scinf[0, 0] = float("inf"); scinf[0, 1] = float("-inf"); scinf[1, :] = 50.0
    validate_case(ck, "inf_saturating", scinf, bias0, 8, True, 1.0, cand)
    # 2e) max captured N
    torch.manual_seed(9)
    scmax = torch.randn(3769, E, dtype=torch.float32, device=DEV)
    biasmax = torch.randn(E, dtype=torch.float32, device=DEV)
    validate_case(ck, "Nmax_3769", scmax, biasmax, 8, True, 1.0, cand)
    # 2f) N=0 no-op (must not crash; outputs empty)
    sc0 = torch.randn(0, E, dtype=torch.float32, device=DEV)
    try:
        bv0, bi0 = run_kernel(base, sc0, bias0, 8, True, 1.0, poison=False)
        ck.check(tuple(bv0.shape) == (0, 8), f"N=0: empty output shape, got {tuple(bv0.shape)}")
        if cand is not None:
            cv0, ci0 = run_kernel(cand, sc0, bias0, 8, True, 1.0, poison=False)
            ck.check(tuple(cv0.shape) == (0, 8), "N=0 candidate empty output")
    except Exception as e:
        ck.check(False, f"N=0 raised: {e}")
    # 2g) renormalize=False path
    validate_case(ck, "renorm_false", scn, bias0, 8, False, 1.0, cand)

    # 3) unsupported parameter must raise (baseline RuntimeCheck) — and candidate must match
    for (ng, tg, why) in [(2, 1, "num_expert_group=2"), (1, 2, "topk_group=2")]:
        torch.manual_seed(0)
        sc_u = torch.randn(4, E, dtype=torch.float32, device=DEV)
        base_raised = False
        try:
            run_kernel(base, sc_u, bias0, 8, True, 1.0, ng=ng, tg=tg, poison=False)
        except Exception:
            base_raised = True
        ck.check(base_raised, f"unsupported {why}: baseline must reject")
        if cand is not None:
            cand_raised = False
            try:
                run_kernel(cand, sc_u, bias0, 8, True, 1.0, ng=ng, tg=tg, poison=False)
            except Exception:
                cand_raised = True
            ck.check(cand_raised == base_raised, f"unsupported {why}: candidate must match baseline reject")

    print(f"\nCORRECTNESS: {ck.npass} checks passed, {ck.fail} failed")
    print("RESULT:", "PASS" if ck.fail == 0 else "FAIL")
    sys.exit(0 if ck.fail == 0 else 1)


if __name__ == "__main__":
    main()
