#!/usr/bin/env python3
"""Isolated benchmark scaffold for `b200_fa4_mha__bf16_head128_total32768`.

Fill `tests/test_correctness.py` first. This script reuses its cases, baseline,
and candidate callables, then appends summary rows to `benchmark.csv`.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


KERNEL_SLUG = "b200_fa4_mha__bf16_head128_total32768"
KERNEL_DIR = Path(__file__).resolve().parent


def _load_correctness_module():
    test_py = KERNEL_DIR / "tests" / "test_correctness.py"
    spec = importlib.util.spec_from_file_location("kda_correctness_scaffold", test_py)
    assert spec is not None and spec.loader is not None, test_py
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sync() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_call(fn: Callable[[dict[str, Any]], Any], case: dict[str, Any], *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn(case)
    _sync()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn(case)
        _sync()
        samples.append((time.perf_counter() - start) * 1e6)
    return samples


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)

    def pct(p: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[index]

    return {
        "median_us": statistics.median(ordered),
        "mean_us": statistics.mean(ordered),
        "std_us": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "min_us": ordered[0],
        "p20_us": pct(0.20),
        "p80_us": pct(0.80),
        "p95_us": pct(0.95),
        "max_us": ordered[-1],
    }


def _case_tflops(case: dict[str, Any], latency_us: float) -> float:
    flops = case.get("flops")
    if flops is None:
        batch = case.get("batch")
        seqlen = case.get("seqlen")
        heads = case.get("num_heads", 16)
        head_dim = case.get("head_dim", 128)
        if batch is None or seqlen is None:
            return float("nan")
        effective_qk = seqlen * seqlen
        if case.get("causal", False):
            effective_qk = seqlen * (seqlen + 1) / 2
        # Approximate attention forward FLOPs: QK and PV matmuls.
        flops = 4.0 * batch * heads * effective_qk * head_dim
    return float(flops) / (latency_us * 1e-6) / 1e12


def _geom_mean(values: list[float]) -> float:
    cleaned = [v for v in values if math.isfinite(v) and v > 0]
    if not cleaned:
        return float("nan")
    return math.exp(sum(math.log(v) for v in cleaned) / len(cleaned))


def main() -> int:
    correctness = _load_correctness_module()
    cases = correctness.make_cases()
    if not cases:
        raise SystemExit("No benchmark cases. Fill tests/test_correctness.py first.")

    candidate_tflops = []
    baseline_tflops = []
    csv_path = KERNEL_DIR / "benchmark.csv"
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        for case in cases:
            warmup = int(case.get("warmup", 20))
            iters = int(case.get("iters", 100))
            baseline_samples = _time_call(correctness.baseline, case, warmup=warmup, iters=iters)
            candidate_samples = _time_call(correctness.candidate, case, warmup=warmup, iters=iters)
            b = _summary(baseline_samples)
            c = _summary(candidate_samples)
            b_tflops = _case_tflops(case, b["mean_us"])
            c_tflops = _case_tflops(case, c["mean_us"])
            baseline_tflops.append(b_tflops)
            candidate_tflops.append(c_tflops)
            now = datetime.now(timezone.utc).isoformat()
            case_name = case.get("shape", case.get("name", "unknown"))
            writer.writerow([
                now,
                case.get("candidate", "baseline_vs_candidate"),
                case_name,
                "mean_tflops",
                f"{b_tflops:.6f}",
                f"{c_tflops:.6f}",
                f"{(c_tflops / b_tflops):.6f}x" if b_tflops and math.isfinite(b_tflops) else "",
                (
                    f"candidate_mean_us={c['mean_us']:.6f} std={c['std_us']:.6f} "
                    f"median={c['median_us']:.6f} p20={c['p20_us']:.6f} "
                    f"p80={c['p80_us']:.6f} p95={c['p95_us']:.6f} "
                    f"min={c['min_us']:.6f} max={c['max_us']:.6f} "
                    f"iters={iters} slug={KERNEL_SLUG}"
                ),
            ])
            print(case_name, "baseline_tflops", b_tflops, "candidate_tflops", c_tflops)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            "geomean",
            "all_configured_cases",
            "geomean_tflops",
            f"{_geom_mean(baseline_tflops):.6f}",
            f"{_geom_mean(candidate_tflops):.6f}",
            "",
            f"slug={KERNEL_SLUG}",
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
