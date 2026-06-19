#!/usr/bin/env python3
"""Correctness: candidate CUDA kernel vs PyTorch baseline (golden_forward math)
on every workload row, contract tolerances (see ../../docs/tilert_correctness_contract.md)."""
import json, os, sys, torch
BENCH = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BENCH)
import adapter
def main():
    dev = torch.device("cuda")
    rows = json.load(open(os.path.join(BENCH, "workloads.json")))
    ok = True
    for r in rows:
        inp = adapter.make_inputs(r["shapes"], dev)
        ref = adapter.call_baseline(inp).float()
        out = adapter.call_candidate(inp).float()
        assert torch.isfinite(out).all(), f"{r['id']}: non-finite"
        rel = (out - ref).norm() / (ref.norm() + 1e-9)
        passed = rel < r.get("rtol", 0.02)
        ok &= passed
        print(f"[{r['id']}] rel={rel:.2e} {'OK' if passed else 'FAIL'}")
    print("CORRECTNESS", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
if __name__ == "__main__": main()
