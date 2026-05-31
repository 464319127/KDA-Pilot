#!/usr/bin/env python3
"""Cold-cache roofline: measure the true HBM-bound latency for the large shapes.

The repeated-launch benchmark leaves Q/K resident in B200's large L2 (so NCU
reports low DRAM%); this script flushes L2 (memset a >L2 scratch buffer) before
each timed launch so the kernel reads Q/K from HBM. It reports achieved DRAM
bandwidth vs an explicit bytes-moved model for baseline and candidate.

Run on ion-b200 inside sglang_bbuf with an idle GPU pinned:
  CUDA_VISIBLE_DEVICES=0 KDA_RUN_CORRECTNESS=1 python roofline_cold.py
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
from pathlib import Path

import torch

KERNEL_DIR = Path(__file__).resolve().parent
SAMPLES = 60
FLUSH_MB = 384  # > B200 L2 (~100-126 MB) to evict the working set
LARGE_CASES = ["qwen__4096", "joyai_edit__7904", "qwen_edit__8424", "zimage__4128"]


def _load_correctness():
    spec = importlib.util.spec_from_file_location(
        "kdac", os.path.join(KERNEL_DIR, "tests", "test_correctness.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    m = _load_correctness()
    sys.path.insert(0, str(KERNEL_DIR / "src"))
    import wrapper

    wrapper.build()
    from sglang.jit_kernel.diffusion.qknorm_rope import fused_inplace_qknorm_rope as baseline_fn

    candidate_fn = wrapper.fused_inplace_qknorm_rope
    scratch = torch.empty(FLUSH_MB * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    peak_bw_tbs = 8.0  # B200 HBM3e approx

    print(f"# cold-cache roofline; L2 flush={FLUSH_MB}MB; samples={SAMPLES}; peak~{peak_bw_tbs} TB/s")
    print(f"# {'case':18s} {'impl':9s} {'cold_us':>9s} {'bytes_MB':>9s} {'BW_TBs':>7s} {'%peak':>6s}")
    for name in LARGE_CASES:
        case = next(c for c in m.make_cases() if c["name"] == name)
        inp = m.build_inputs(case)
        n, h, d = case["num_tokens"], case["num_heads"], case["head_dim"]
        # bytes moved: read+write q and k (bf16) + one cos_sin row per token (f32, L2-reused intra-launch)
        bytes_moved = 4 * n * h * d * 2 + n * case["rope_dim"] * 4
        bytes_mb = bytes_moved / 1e6
        qw, kw, csc, pos = inp["q_weight"], inp["k_weight"], inp["cos_sin_cache"], inp["positions"]
        is_neox, eps, rope_dim = inp["is_neox"], inp["eps"], inp["rope_dim"]
        q0, k0 = inp["q"], inp["k"]  # pristine, never mutated
        qb, kb = q0.clone(), k0.clone()
        for label, fn in (("baseline", baseline_fn), ("candidate", candidate_fn)):
            def run():
                fn(qb, kb, qw, kw, csc, pos, is_neox=is_neox, eps=eps, rope_dim=rope_dim)
            for _ in range(10):
                qb.copy_(q0); kb.copy_(k0); run()
            torch.cuda.synchronize()
            samples = []
            for _ in range(SAMPLES):
                qb.copy_(q0); kb.copy_(k0)  # pristine reset, then evict so the kernel reads cold
                scratch.zero_()             # evict L2 (q/k written by the reset + scratch)
                torch.cuda.synchronize()
                start.record()
                run()
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end) * 1000.0)  # us
            med = statistics.median(samples)
            bw = bytes_moved / (med * 1e-6) / 1e12  # TB/s
            print(f"  {name:18s} {label:9s} {med:9.2f} {bytes_mb:9.1f} {bw:7.2f} {100*bw/peak_bw_tbs:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
