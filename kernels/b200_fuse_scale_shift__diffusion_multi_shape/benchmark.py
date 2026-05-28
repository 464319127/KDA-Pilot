#!/usr/bin/env python3
"""Isolated benchmark scaffold for ``b200_fuse_scale_shift__diffusion_multi_shape``.

Fill ``tests/test_correctness.py`` first. This script reuses its cases, baseline,
and candidate callables, then appends summary rows to ``benchmark.csv``.
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


KERNEL_SLUG = "b200_fuse_scale_shift__diffusion_multi_shape"
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
        "p10_us": pct(0.10),
        "p90_us": pct(0.90),
        "p95_us": pct(0.95),
        "max_us": ordered[-1],
    }


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

    speedups = []
    csv_path = KERNEL_DIR / "benchmark.csv"
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        for case in cases:
            warmup = int(case.get("warmup", 25))
            iters = int(case.get("iters", 100))
            baseline_samples = _time_call(correctness.baseline, case, warmup=warmup, iters=iters)
            candidate_samples = _time_call(correctness.candidate, case, warmup=warmup, iters=iters)
            b = _summary(baseline_samples)
            c = _summary(candidate_samples)
            speedup = (b["median_us"] / c["median_us"]) if c["median_us"] > 0 else float("nan")
            speedups.append(speedup)
            now = datetime.now(timezone.utc).isoformat()
            case_name = case.get("name", case.get("shape", "unknown"))
            writer.writerow([
                now,
                case.get("candidate", "baseline_vs_candidate"),
                case_name,
                "median_us",
                f"{b['median_us']:.6f}",
                f"{c['median_us']:.6f}",
                f"{speedup:.6f}x" if math.isfinite(speedup) else "",
                (
                    f"baseline_mean_us={b['mean_us']:.6f} cand_mean_us={c['mean_us']:.6f} "
                    f"cand_p10={c['p10_us']:.6f} cand_p90={c['p90_us']:.6f} "
                    f"iters={iters} slug={KERNEL_SLUG}"
                ),
            ])
            print(case_name, "speedup_x", speedup)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            "geomean",
            "all_configured_shapes",
            "geomean_speedup_x",
            "",
            "",
            f"{_geom_mean(speedups):.6f}x",
            f"slug={KERNEL_SLUG}",
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
