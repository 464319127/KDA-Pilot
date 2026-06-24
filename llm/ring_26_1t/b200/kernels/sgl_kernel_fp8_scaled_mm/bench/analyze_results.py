#!/usr/bin/env python3
"""Summarize a full-grid results.jsonl for the AC-6/AC-5/AC-8 verdict:
- equal-weight / call-weighted / time-weighted geomean over production rows
- per-regime geomean
- covered M=1 shapes: per-shape median + p10/p90 + significance (candidate p90 <
  baseline p10 == non-overlapping distributions) + significant-regression check
- fallback overhead for uncovered (route-0) production shapes (candidate goes
  through the predicate + baseline_impl): median/max overhead, prove <1%

Usage: analyze_results.py <results.jsonl>
"""
import json
import math
import statistics
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
res_path = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "results_full.jsonl")

rows = []
for line in open(res_path):
    line = line.strip()
    if not line:
        continue
    try:
        o = json.loads(line)
    except Exception:
        continue
    if isinstance(o, dict) and o.get("status") == "PASSED" and "speedup" in o and "baseline" in o:
        rows.append(o)


def covered(wl):  # mirrors covers_m1_gemv shape conditions
    s = wl["shapes"]
    M, K, N = s["M"], s["K"], s["N"]
    return M == 1 and not (K >= 4096 and N >= 3072)


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def wgeomean(rs, weight):
    num = den = 0.0
    for r in rs:
        sp, w = r["speedup"], weight(r)
        if sp > 0 and w > 0:
            num += w * math.log(sp)
            den += w
    return math.exp(num / den) if den > 0 else float("nan")


prod = [r for r in rows if r.get("production")]
calls = lambda r: r["workload"].get("calls", 0) or 0
btime = lambda r: (r["workload"].get("calls", 0) or 0) * r["baseline"]["median_us"]

print(f"# Full-grid analysis ({len(prod)} production rows)\n")
print(f"equal-weight geomean : {geomean([r['speedup'] for r in prod]):.4f}")
print(f"call-weighted geomean: {wgeomean(prod, calls):.4f}")
print(f"time-weighted geomean: {wgeomean(prod, btime):.4f}")

cov = [r for r in prod if covered(r["workload"])]
unc = [r for r in prod if not covered(r["workload"])]
print(f"\ncovered(M=1) geomean : {geomean([r['speedup'] for r in cov]):.4f}  (n={len(cov)})")

print("\n## Covered M=1 shapes — significance (sig_win = speedup>=1.10 AND candidate p90 < baseline p10)")
n_sig = n_reg = 0
for r in sorted(cov, key=lambda r: -r["speedup"]):
    b, c, sp = r["baseline"], r["candidate"], r["speedup"]
    sig_win = sp >= 1.10 and c["p90_us"] < b["p10_us"]
    sig_reg = sp < 1.0 and b["p90_us"] < c["p10_us"]
    n_sig += int(sig_win)
    n_reg += int(sig_reg)
    print(f"  {r['id']:<22} sp={sp:.3f}  base={b['median_us']:.2f}us(p10 {b['p10_us']:.2f}/p90 {b['p90_us']:.2f})"
          f"  cand={c['median_us']:.2f}us(p10 {c['p10_us']:.2f}/p90 {c['p90_us']:.2f})  sig_win={sig_win}")
print(f"  => {n_sig}/{len(cov)} covered shapes are significant >=1.10 wins; significant regressions: {n_reg}")

ov = [(r["candidate"]["median_us"] / r["baseline"]["median_us"] - 1.0)
      for r in unc if r["baseline"]["median_us"] > 0]
print(f"\n## Fallback overhead over {len(unc)} uncovered production shapes (route-0 candidate vs direct baseline)")
print(f"  median={statistics.median(ov)*100:+.3f}%  mean={statistics.mean(ov)*100:+.3f}%"
      f"  max={max(ov)*100:+.3f}%  min={min(ov)*100:+.3f}%")
print(f"  |overhead|<1%: {sum(1 for x in ov if abs(x) < 0.01)/len(ov)*100:.1f}% of shapes;"
      f"  median |overhead| = {statistics.median([abs(x) for x in ov])*100:.3f}%")

from collections import defaultdict
reg_g = defaultdict(list)
for r in prod:
    reg_g[r["workload"].get("regime", "?")].append(r["speedup"])
print("\n## Per-regime geomean")
for k in sorted(reg_g):
    print(f"  {k:<14} geomean={geomean(reg_g[k]):.4f}  (n={len(reg_g[k])})")
