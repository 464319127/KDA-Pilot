"""Controlled launch-overhead floor probe for topk_sigmoid.

Times an empty kernel (topk_sigmoid_noop) launched at the SAME grid/block the candidate fast
path uses, for representative token counts. This bounds the irreducible per-call launch +
dispatch overhead, which is the dominant cost for the N=1 decode regime — it tells us how much
of the baseline-vs-candidate gap a single-launch fused kernel can actually recover (vs the
baseline's two launches + workspace allocation).

Same CUDA-event timing + inner-loop amplification style as benchmark.py. Run on the target GPU:
  python3 bench/floor_probe.py
"""

from __future__ import annotations

import statistics

import torch

import build_ext

PROBE_N = [1, 16, 80, 1579, 10207, 16883]
WARMUP = 20
TRIALS = 7
TARGET_US = 1000.0
INNER_MAX = 4096


def _calibrate_inner(fn) -> int:
    inner = 1
    while inner < INNER_MAX:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        if start.elapsed_time(end) * 1000.0 >= TARGET_US:
            break
        inner *= 2
    return inner


def time_noop(n: int, device) -> dict:
    gating = torch.randn((n, 288), dtype=torch.float32, device=device)
    fn = lambda: build_ext.noop(gating)
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    inner = _calibrate_inner(fn)
    samples = []
    for _ in range(TRIALS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / inner)  # us per launch
    samples.sort()
    return {"n": n, "inner": inner, "median_us": statistics.median(samples),
            "min_us": samples[0], "p90_us": samples[min(len(samples) - 1, int(0.9 * len(samples)))]}


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 2
    device = torch.device("cuda")
    print(f"launch-floor probe (empty kernel @ candidate grid), GPU={torch.cuda.get_device_name(0)}")
    print(f"{'N':>8} {'inner':>7} {'median_us':>12} {'min_us':>10} {'p90_us':>10}")
    for n in PROBE_N:
        r = time_noop(n, device)
        print(f"{r['n']:>8} {r['inner']:>7} {r['median_us']:>12.4f} {r['min_us']:>10.4f} {r['p90_us']:>10.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
