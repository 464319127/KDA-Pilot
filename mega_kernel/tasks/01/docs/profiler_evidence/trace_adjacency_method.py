import collections
import glob
import gzip
import json

f = sorted(glob.glob("/scratch/glm52_blog_bench/profiles/task01_ba/*TP-0*.trace.json.gz"))[-1]
print("trace:", f)
events = json.load(gzip.open(f, "rt"))
events = events["traceEvents"] if isinstance(events, dict) else events
ks = sorted((e["ts"], e["dur"], e["name"]) for e in events
            if e.get("ph") == "X" and e.get("cat") == "kernel")
after = collections.Counter()
before = collections.Counter()
n = 0
for i, (ts, dur, name) in enumerate(ks):
    if "AllreduceFusion" in name or "ArFusedNormConst" in name:
        n += 1
        if i + 1 < len(ks):
            after[ks[i + 1][2][:110]] += 1
        if i > 0:
            before[ks[i - 1][2][:110]] += 1
print(f"fused-AR instances: {n}")
print("== kernels FOLLOWING the fused AR ==")
for nm, c in after.most_common(8):
    print(f"  {c:6d}  {nm}")
print("== kernels PRECEDING the fused AR ==")
for nm, c in before.most_common(8):
    print(f"  {c:6d}  {nm}")
