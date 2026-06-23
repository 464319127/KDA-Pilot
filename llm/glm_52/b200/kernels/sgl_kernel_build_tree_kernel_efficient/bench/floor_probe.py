#!/usr/bin/env python3
"""Controlled same-process diagnostics (secondary evidence) for build_tree.

Section A: per-bs controlled A/B probe for ALL bs 1..10 (incl. 7 and 9) — both
sides timed back-to-back in ONE process (one GPU-clock state) to remove the
cross-subprocess clock noise of the official harness. Each measured invocation
draws fresh -1 retrive_next_token / retrive_next_sibling rows from a 2D ring (so
the baseline always takes its if-branch, no reuse artifact); tree_mask / positions
/ retrive_index are idempotent under this op and shared. Reports floor / baseline /
candidate medians + p10/p90 and a strict non-overlap "clean_win" flag.

Section B: wrapper-inclusive diagnostic (DEC-1 secondary; task13) — times the real
per-call wrapper cost the captured callsite pays for the op's outputs
(tree_mask.fill_(True) + retrieve_buf = torch.full((3,bs,nv), -1) + positions =
torch.empty(...)) PLUS the op, vs the op alone, at representative shapes. Quantifies
the prefill that a future wrapper-fusion patch (out of promotion scope) could remove.

Run: CUDA_VISIBLE_DEVICES=6 python bench/floor_probe.py
"""

import math
import statistics
import sys

sys.path.insert(0, "bench")
import torch  # noqa: E402

import build_ext  # noqa: E402
from benchmark import _calibrate_inner, _cuda_event_time_us  # noqa: E402

dev = torch.device("cuda")
NV = 2
RING = 65536  # > per-bs (baseline+candidate) measure() invocations; hard-asserted below


def measure(fn, trials=31):
    for _ in range(50):
        fn()  # warm clocks before calibration (avoid cold-start)
    torch.cuda.synchronize()
    inner = _calibrate_inner(fn, inner_min=1, inner_max=4096, target_sample_us=1000.0)
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = sorted(_cuda_event_time_us(fn, inner)[0] for _ in range(trials))
    return statistics.median(s), s[int(0.1 * (len(s) - 1))], s[int(0.9 * (len(s) - 1))], inner


def probe_fns(bs, L=64):
    T = 2 * L * bs + 4 * bs
    vsl = torch.full((bs,), L, dtype=torch.int64, device=dev)
    pl = torch.empty((bs, 0), dtype=torch.int64, device=dev)
    si = torch.zeros((bs, 1), dtype=torch.int64, device=dev)
    tm = torch.full((T,), True, dtype=torch.bool, device=dev)        # idempotent, shared
    pos = torch.full((bs * NV,), -1, dtype=torch.int64, device=dev)  # idempotent, shared
    ri = torch.full((bs * NV,), -1, dtype=torch.int64, device=dev)   # idempotent, shared
    nt = torch.full((RING, bs * NV), -1, dtype=torch.int64, device=dev)  # fresh per call
    ns = torch.full((RING, bs * NV), -1, dtype=torch.int64, device=dev)  # fresh per call
    c = [0]

    def _row():
        i = c[0]
        if i >= RING:
            raise RuntimeError("probe ring wrapped; increase RING")
        c[0] += 1
        return nt[i], ns[i]

    def base():
        a, b = _row()
        build_ext.baseline(pl, si, vsl, tm, pos, ri, a, b, 1, 1, 2, 0)

    def cand():
        a, b = _row()
        build_ext.candidate(pl, si, vsl, tm, pos, ri, a, b, 1, 1, 2, 0)

    return {"noop": lambda: build_ext.noop(vsl, 2), "baseline": base, "candidate": cand}


def section_a():
    print("== Section A: per-bs controlled A/B probe (all bs 1..10) ==")
    print(f"{'bs':>3} {'floor':>7} {'base':>7} {'b_p10':>6} {'b_p90':>6} "
          f"{'cand':>7} {'c_p10':>6} {'c_p90':>6} {'spd':>6} {'clean_win':>9}")
    geo = []
    for bs in range(1, 11):
        f = probe_fns(bs)
        r = {n: measure(f[n]) for n in ("noop", "baseline", "candidate")}
        fl = r["noop"][0]
        bm, bp10, bp90 = r["baseline"][:3]
        cm, cp10, cp90 = r["candidate"][:3]
        spd = bm / cm
        clean = cp90 < bp10           # candidate strictly faster, non-overlapping p10/p90
        regress = cp10 > bp90         # candidate strictly slower, non-overlapping
        geo.append(spd)
        tag = "WIN" if clean else ("REGRESS" if regress else "tie")
        print(f"{bs:>3} {fl:>7.3f} {bm:>7.3f} {bp10:>6.3f} {bp90:>6.3f} "
              f"{cm:>7.3f} {cp10:>6.3f} {cp90:>6.3f} {spd:>6.3f} {tag:>9}")
    print(f"geomean speedup (bs 1..10) = {math.exp(sum(math.log(x) for x in geo) / len(geo)):.4f}")


def wrapper_fns(bs, T):
    L_sum = (T - 4 * bs) // 2
    base = L_sum // bs
    vsl = torch.full((bs,), base, dtype=torch.int64, device=dev)
    vsl[0] += L_sum - base * bs
    pl = torch.empty((bs, 0), dtype=torch.int64, device=dev)
    si = torch.zeros((bs, 1), dtype=torch.int64, device=dev)
    tm = torch.full((T,), True, dtype=torch.bool, device=dev)
    rbuf = torch.full((3, bs, NV), -1, dtype=torch.int64, device=dev)
    pos = torch.empty((bs * NV,), dtype=torch.int64, device=dev)

    def op_only():
        build_ext.candidate(pl, si, vsl, tm, pos, rbuf[0], rbuf[1], rbuf[2], 1, 1, 2, 0)

    def wrapper_inclusive():
        # exactly what the captured Python callsite pays per call for the op outputs
        tm.fill_(True)
        rb = torch.full((3, bs, NV), -1, dtype=torch.int64, device=dev)
        p = torch.empty((bs * NV,), dtype=torch.int64, device=dev)
        build_ext.candidate(pl, si, vsl, tm, p, rb[0], rb[1], rb[2], 1, 1, 2, 0)

    return op_only, wrapper_inclusive


def section_b():
    print("\n== Section B: wrapper-inclusive diagnostic (DEC-1 secondary, NOT promoted) ==")
    print(f"{'bs':>3} {'T':>7} {'op_only':>8} {'wrap_incl':>10} {'prefill_add':>12} {'op_frac':>8}")
    for bs, T in ((4, 2246), (10, 5382), (10, 11626)):
        op_only, wrap = wrapper_fns(bs, T)
        o = measure(op_only)[0]
        w = measure(wrap)[0]
        print(f"{bs:>3} {T:>7} {o:>8.3f} {w:>10.3f} {w - o:>12.3f} {o / w:>8.3f}")


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    section_a()
    section_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
