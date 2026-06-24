"""Candidate-only decode latency measurement.

The recovered baseline's decode (small-token) path is UB for num_experts=128 and is not safely
benchmarkable (see docs/baseline_source.md), so the standard A/B `benchmark.py` (which times BOTH
sides) cannot produce decode numbers. This script reports the CANDIDATE's absolute decode latency
using the same timing methodology as the template (CUDA-event timing, inner-loop amplification,
median over trials). It deliberately reports NO speedup ratio for decode — the baseline cannot run
there. Run on the idle target GPU:

    CUDA_VISIBLE_DEVICES=4 python bench/bench_decode_candidate.py
"""

from __future__ import annotations

import statistics
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _jit_build import candidate_module  # noqa: E402

NUM_EXPERTS = 128
TOPK = 5
DECODE_M = [1, 7, 12, 16, 24, 32, 44, 61, 79, 512]


def median_us(mod, M: int, inner: int = 2000, trials: int = 9, warmup: int = 20) -> float:
    g = torch.Generator(device="cuda"); g.manual_seed(M)
    inp = torch.randn((M, NUM_EXPERTS), dtype=torch.float32, device="cuda", generator=g).contiguous()
    bias = torch.randn((NUM_EXPERTS,), dtype=torch.float32, device="cuda", generator=g).contiguous()
    out = torch.empty((M, TOPK), dtype=torch.float32, device="cuda")
    idx = torch.empty((M, TOPK), dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        mod.moe_fused_gate(inp, bias, out, idx, TOPK, 0, 1, True, 2.0, True)
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            mod.moe_fused_gate(inp, bias, out, idx, TOPK, 0, 1, True, 2.0, True)
        e.record(); torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1000.0 / inner)  # us per call
    return statistics.median(samples)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; run on the target GPU.")
        return 2
    cand = candidate_module()
    print("candidate-only decode latency (us/call); baseline decode is UB/unbenchmarkable")
    for M in DECODE_M:
        print(f"  M={M:4d}: {median_us(cand, M):.3f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
