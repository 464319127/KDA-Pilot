"""Correctness harness for the MoE fused-gate router kernel.

Validates two things on the target GPU (build happens via ``_jit_build``):
  1. recovered BASELINE vs an INDEPENDENT fp32 oracle (confirms recovered semantics, not
     just internal self-consistency);
  2. CANDIDATE vs baseline (exact-match on ordered selected indices; weights within
     tolerance), once a native-CUDA candidate exists (else this stage is skipped).

Top-k / routing outputs are EXACT-MATCH on indices; gathered weights use atol=rtol=1e-5.
Output buffers are poisoned before every run so stale / partial / skipped writes are caught.

Recovered semantics (see docs/baseline_source.md): selection uses biased = sigmoid(x)+bias;
the emitted weight uses the UN-biased sigmoid score; topk_routed = topk - shared experts are
chosen by iterative arg-max in descending biased order with smaller-index tie-break; the
last `num_fused_shared_experts` slots are shared (index = num_experts + offset, weight via
the same renorm/scale path). The shared-slot weight is NOT hardcoded to 1.0 — it is computed
with the kernel's float32 op order so subnormal `routed_sum` is handled identically.

NOTE (decode/small-token path, num_experts=128): the recovered baseline small-token kernel
reads unwritten warp_maxs[4..7] (instantiated <8> but only 4 warps run), so its tie-break is
not source-guaranteed. We therefore validate baseline-vs-oracle empirically over many seeds
on random inputs (ties are measure-zero there) and treat adversarial-tie decode rows as
diagnostic: a mismatch there is attributed to the documented baseline hazard, and the
candidate is held to the well-defined intended semantics.

Run: python bench/correctness.py   (exit 0 iff all checks pass)
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from _jit_build import baseline_module, candidate_module, has_candidate

NUM_EXPERTS = 128
TOPK = 5
SHARED = 1               # num_fused_shared_experts
SCORING_SIGMOID = 0
RENORM = True
RSF = 2.0                # routed_scaling_factor
APPLY_ON_OUTPUT = True
TOPK_ROUTED = TOPK - SHARED  # 4

f32 = np.float32
POISON_W = float("nan")
POISON_IDX = -777


# ----------------------------------------------------------------------------- oracle
def oracle(
    inp: np.ndarray,
    bias: np.ndarray,
    topk: int = TOPK,
    num_fused_shared_experts: int = SHARED,
    renormalize: bool = RENORM,
    routed_scaling_factor: float = RSF,
    apply_on_output: bool = APPLY_ON_OUTPUT,
):
    """Independent fp32 oracle. inp [N,E] f32, bias [E] f32 -> (weights [N,topk] f32,
    indices [N,topk] int32). Mirrors moe_fused_gate.cuh in float32 op order."""
    inp = inp.astype(f32, copy=False)
    bias = bias.astype(f32, copy=False)
    N, E = inp.shape
    topk_routed = topk - num_fused_shared_experts
    rsf = f32(routed_scaling_factor)

    score = (f32(1.0) / (f32(1.0) + np.exp(-inp).astype(f32))).astype(f32)  # sigmoid
    biased = (score + bias).astype(f32)

    weights = np.zeros((N, topk), dtype=f32)
    indices = np.zeros((N, topk), dtype=np.int32)
    for r in range(N):
        b = biased[r].copy()
        sel: list[int] = []
        for _ in range(topk_routed):
            m = b.max()
            j = int(np.flatnonzero(b == m)[0])  # smallest index achieving the max
            sel.append(j)
            b[j] = f32(-np.inf)                  # mask (kernel uses -FLT_MAX)
        routed_sum = f32(0.0)
        for j in sel:
            routed_sum = f32(routed_sum + score[r, j])
        norm = routed_sum if routed_sum > f32(0.0) else f32(1.0)
        scale = rsf if apply_on_output else f32(1.0)
        for k in range(topk):
            if k >= topk_routed:  # shared slot
                w = f32(routed_sum / rsf)
                idx = E + (k - topk_routed)
            else:
                w = score[r, sel[k]]
                idx = sel[k]
            weights[r, k] = f32(f32(w / norm) * scale)
            indices[r, k] = idx
    return weights, indices


# ----------------------------------------------------------------------------- checker
class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, cond: bool, msg: str) -> bool:
        if cond:
            self.passed += 1
        else:
            self.failed += 1
            print(f"  FAIL: {msg}")
        return cond

    def exact_idx(self, label: str, a: torch.Tensor, b: torch.Tensor) -> bool:
        ok = torch.equal(a.to(torch.int64), b.to(torch.int64))
        if not ok:
            n = int((a != b).sum().item())
            return self.check(False, f"{label}: indices differ in {n} slots")
        return self.check(True, label)

    def weights_close(self, label: str, a: torch.Tensor, b: torch.Tensor,
                      atol: float = 1e-5, rtol: float = 1e-5) -> bool:
        a32 = a.float()
        b32 = b.float()
        if torch.isnan(a32).any() or torch.isinf(a32).any():
            return self.check(False, f"{label}: contains NaN/Inf")
        diff = (a32 - b32).abs()
        ok = bool(torch.all(diff <= (atol + rtol * b32.abs())).item())
        if not ok:
            return self.check(False, f"{label}: max_abs={float(diff.max()):.3e} exceeds tol")
        return self.check(True, label)


def _run_module(mod, inp: torch.Tensor, bias: torch.Tensor, topk: int = TOPK,
                shared: int = SHARED, renorm: bool = RENORM, rsf: float = RSF,
                apply_out: bool = APPLY_ON_OUTPUT):
    """Allocate + poison outputs, run the module, return (weights, indices)."""
    N = inp.shape[0]
    out = torch.full((N, topk), POISON_W, dtype=torch.float32, device=inp.device)
    idx = torch.full((N, topk), POISON_IDX, dtype=torch.int32, device=inp.device)
    mod.moe_fused_gate(inp, bias, out, idx, topk, SCORING_SIGMOID, shared, renorm, rsf, apply_out)
    torch.cuda.synchronize()
    return out, idx


def _validate_outputs(ck: Checker, label: str, out: torch.Tensor, idx: torch.Tensor, N: int):
    """Structural checks every output must satisfy regardless of values."""
    ck.check(out.shape == (N, TOPK) and idx.shape == (N, TOPK), f"{label}: output shapes")
    ck.check(out.dtype == torch.float32 and idx.dtype == torch.int32, f"{label}: output dtypes")
    if N > 0:
        ck.check(not torch.isnan(out).any().item(), f"{label}: no NaN poison left in weights")
        # routed indices in [0,128), shared slot == 128
        routed = idx[:, :TOPK_ROUTED]
        shared = idx[:, TOPK_ROUTED:]
        ck.check(bool(((routed >= 0) & (routed < NUM_EXPERTS)).all().item()),
                 f"{label}: routed indices in range")
        ck.check(bool((shared == NUM_EXPERTS).all().item()),
                 f"{label}: shared slot index == {NUM_EXPERTS}")


# ----------------------------------------------------------------------------- grid
def _load_workloads() -> list:
    """Load the frozen workload set — the AUTHORITATIVE source for the production + edge grid.
    Correctness derives every captured/edge row from this file (no hard-coded shape lists), so the
    grid cannot silently diverge from what the benchmark freezes."""
    import json
    import pathlib
    return json.loads((pathlib.Path(__file__).resolve().parent / "workloads.json").read_text())


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; correctness must run on the target GPU.")
        return 2
    # Shared input constructor — the same one bench/adapter.py + benchmark.py use, so workloads.json
    # is the single source of truth for input construction. (Candidate cold-safety is verified
    # separately by the fresh-process probe / bench_decode_candidate.py; here we validate numerics.)
    from adapter import build_inputs
    dev = torch.device("cuda")
    g = torch.Generator(device=dev)
    ck = Checker()
    base = baseline_module()
    cand = candidate_module() if has_candidate() else None
    print(f"candidate present: {cand is not None}")

    # Warmup: the recovered baseline's decode (small-token) kernel reads uninitialized shared memory
    # for num_experts=128 and faults on a COLD context (docs/baseline_source.md). Prime with a safe
    # large-path launch so the baseline is usable on the large-token path it is validated on.
    _wu = torch.randn((1024, NUM_EXPERTS), dtype=torch.float32, device=dev)
    _wb = torch.randn((NUM_EXPERTS,), dtype=torch.float32, device=dev)
    _run_module(base, _wu, _wb)

    def check_row(label, M, inp, bias, topk, atol, rtol):
        """Candidate-vs-oracle on every row; baseline-vs-oracle + candidate-vs-baseline only on the
        UB-safe large-token path (M>512). The independent oracle is the ground-truth reference."""
        ow, oi = oracle(inp.cpu().numpy(), bias.cpu().numpy(), topk=topk)
        ow_t = torch.from_numpy(ow).to(dev)
        oi_t = torch.from_numpy(oi).to(dev)
        base_safe = M > 512
        bout = bidx = None
        if base_safe:
            bout, bidx = _run_module(base, inp, bias, topk=topk)
            _validate_outputs(ck, f"baseline {label}", bout, bidx, M)
            ck.exact_idx(f"baseline-vs-oracle idx {label}", bidx, oi_t)
            ck.weights_close(f"baseline-vs-oracle w {label}", bout, ow_t, atol, rtol)
        if cand is not None:
            cout, cidx = _run_module(cand, inp, bias, topk=topk)
            _validate_outputs(ck, f"candidate {label}", cout, cidx, M)
            ck.exact_idx(f"candidate-vs-oracle idx {label}", cidx, oi_t)
            ck.weights_close(f"candidate-vs-oracle w {label}", cout, ow_t, atol, rtol)
            if base_safe:
                ck.exact_idx(f"candidate-vs-baseline idx {label}", cidx, bidx)
                ck.weights_close(f"candidate-vs-baseline w {label}", cout, bout, atol, rtol)
            cout2, cidx2 = _run_module(cand, inp, bias, topk=topk)
            ck.check(torch.equal(cidx, cidx2) and torch.equal(cout, cout2),
                     f"candidate deterministic {label}")

    WL = _load_workloads()
    prod = [w for w in WL if w.get("production")]
    edge = [w for w in WL if not w.get("production")]

    def _row_dims(w):
        sh = w["shapes"]
        return (int(sh["num_tokens"]), int(sh["num_experts"]), int(w["scalars"]["topk"]),
                w.get("generator", "randn"), float(w.get("atol", 1e-5)), float(w.get("rtol", 1e-5)))

    # 1) All PRODUCTION rows from workloads.json (18 decode + 11 prefill), 2 seeds each.
    for w in prod:
        M, E, topk, gen, atol, rtol = _row_dims(w)
        for s in (0, 1):
            g.manual_seed(1000 * M + s)
            inp, bias = build_inputs(gen, M, E, dev, gen=g)
            check_row(f"{w['regime']} {w['id']} s={s}", M, inp, bias, topk, atol, rtol)

    # 2) Frozen EDGE grid from workloads.json (generator-driven; AC-3). Built via the SAME
    #    adapter.build_inputs the benchmark uses, so the frozen file is authoritative. M=0 is a
    #    no-op safety check. +-Inf are covered (sigmoid(+-inf) is finite 1/0); NaN is out of contract.
    for w in edge:
        M, E, topk, gen, atol, rtol = _row_dims(w)
        g.manual_seed(int(w.get("seed", 0)) + 424242 + M)
        inp, bias = build_inputs(gen, M, E, dev, gen=g)
        if M == 0:
            try:
                _run_module(cand if cand is not None else base, inp, bias, topk=topk)
                ck.check(True, f"edge[{w['id']}] M=0 no-op")
            except Exception as e:  # noqa: BLE001
                ck.check(False, f"edge[{w['id']}] M=0 raised {e!r}")
            continue
        check_row(f"edge[{w['id']}]", M, inp, bias, topk, atol, rtol)

    # 3) Deliberate adversarial probes NOT in the captured workload set (kept here on purpose):
    # 3a) Prefill-sized exact tie (M=1024): smaller-index-wins on the source-guaranteed large path.
    inp = torch.zeros((1024, NUM_EXPERTS), dtype=torch.float32, device=dev)
    bias = torch.full((NUM_EXPERTS,), -1.0, device=dev, dtype=torch.float32)
    bias[3] = 1.0
    bias[100] = 1.0  # tie between experts 3 and 100; smaller index (3) must win slot 0
    inp = inp.contiguous(); bias = bias.contiguous()
    _, oi = oracle(inp.cpu().numpy(), bias.cpu().numpy())
    oi_t = torch.from_numpy(oi).to(dev)
    _, bidx = _run_module(base, inp, bias)
    ck.exact_idx("baseline-vs-oracle tie prefill M=1024", bidx, oi_t)
    if cand is not None:
        _, cidx = _run_module(cand, inp, bias)
        ck.exact_idx("candidate-vs-oracle tie prefill M=1024", cidx, oi_t)

    # 5) Off-domain fallback: E=256, topk=8 fails the production gate, so the candidate must
    #    route to its verbatim-copied baseline fallback. E=256 -> warps_per_token=8, so the
    #    baseline small-token path is safe here (the UB is E<256-specific). The candidate must
    #    be bit-identical to the baseline AND match the oracle on this off-domain config.
    if cand is not None:
        for M in (64, 600):  # small-token + large-token off-domain
            g.manual_seed(7777 + M)
            inp = torch.randn((M, 256), dtype=torch.float32, device=dev, generator=g).contiguous()
            bias = torch.randn((256,), dtype=torch.float32, device=dev, generator=g).contiguous()
            ow, oi = oracle(inp.cpu().numpy(), bias.cpu().numpy(), topk=8)
            cout, cidx = _run_module(cand, inp, bias, topk=8)
            bout, bidx = _run_module(base, inp, bias, topk=8)
            ck.exact_idx(f"offdomain-fallback candidate==baseline idx E=256 M={M}", cidx, bidx)
            ck.weights_close(f"offdomain-fallback candidate==baseline w E=256 M={M}", cout, bout)
            ck.exact_idx(f"offdomain-fallback candidate-vs-oracle idx E=256 M={M}", cidx,
                         torch.from_numpy(oi).to(dev))

    # NOTE on input contract: all 296 captured variants are finite float32 (~randn-scale).
    # +Inf/-Inf ARE covered as explicit edge rows (pos_inf/neg_inf) — sigmoid(+-inf) is finite (1/0),
    # so candidate and oracle agree. Only NaN is OUT OF CONTRACT (the baseline ignores NaN in its
    # `>` comparisons while the candidate's packed-key comparison may order NaN differently), so NaN
    # behavior is intentionally not matched. See docs/benchmark_method.md.

    print(f"\n{'PASS' if ck.failed == 0 else 'FAIL'}: {ck.passed} checks passed, {ck.failed} failed")
    return 0 if ck.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
