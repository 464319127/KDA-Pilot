#!/usr/bin/env python3
"""Benchmark the fused in-place QK-Norm + RoPE candidate vs the SGLang baseline.

Fair in-place timing on a verified-idle B200:
  * CUDA-event timing of ONE kernel invocation per sample (no batching); the
    per-call latency is the median over many such single-invocation samples;
  * the in-place Q/K work buffers are reset from a pristine copy and synchronized
    immediately before each timed sample's start.record() (the plan requires
    re-initializing the inputs between timed iterations); the reset is excluded
    from the timed region;
  * modules (CUDA extension + SGLang baseline JIT) are warmed before timing;
  * GPU utilization/memory are recorded before the run and after settling back to
    idle, for provenance.

Reports median / mean / std / min / p10 / p90 per shape and the geometric mean
of per-shape median-latency speedups, and appends rows (with host / GPU id /
GPU model / commit / command provenance) to ``benchmark.csv``.

Run on ion-b200 inside sglang_bbuf with an idle GPU pinned, e.g.:
  CUDA_VISIBLE_DEVICES=0 KDA_RUN_CORRECTNESS=1 python benchmark.py
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

KERNEL_DIR = Path(__file__).resolve().parent
CSV_PATH = KERNEL_DIR / "benchmark.csv"
# One kernel invocation per CUDA-event sample, with a pristine Q/K reset before
# each timed sample (the plan requires re-initializing the in-place inputs
# between timed iterations); many samples keep the tiny-shape median stable.
SAMPLES = 200
WARMUP = 30

CSV_FIELDS = [
    "timestamp", "host", "gpu_id", "gpu_name", "case", "bucket", "num_tokens",
    "num_heads", "head_dim", "rope_dim", "is_neox", "eps", "impl",
    "median_us", "mean_us", "std_us", "min_us", "p10_us", "p90_us",
    "dispatch_path", "speedup_vs_baseline", "sglang_version", "candidate_src",
    "kp_commit", "gpu_util_before", "gpu_mem_before_mb", "gpu_util_after",
    "gpu_mem_after_mb", "command",
]


def _gpu_state() -> tuple[str, str]:
    """(utilization%, memory_used_MB) for the pinned physical GPU, via nvidia-smi."""
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-i", phys], text=True).strip()
        util, mem = (x.strip() for x in out.split(","))
        return util, mem
    except Exception:  # noqa: BLE001
        return "?", "?"


def _load_correctness_module():
    test_py = KERNEL_DIR / "tests" / "test_correctness.py"
    spec = importlib.util.spec_from_file_location("kda_correctness", test_py)
    assert spec is not None and spec.loader is not None, test_py
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percentile(ordered: list[float], p: float) -> float:
    if not ordered:
        return float("nan")
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[idx]


def _time_impl(run: Callable[[], None], reset: Callable[[], None]) -> dict[str, float]:
    """Per-call latency (us): one kernel invocation per CUDA-event sample, with a
    pristine in-place Q/K reset (excluded from the timed region) before each."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP):
        reset()
        run()
    torch.cuda.synchronize()
    samples_us: list[float] = []
    for _ in range(SAMPLES):
        reset()
        torch.cuda.synchronize()  # ensure the reset copy is done before timing
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)  # ms -> us, single call
    ordered = sorted(samples_us)
    return {
        "median_us": statistics.median(ordered),
        "mean_us": statistics.fmean(ordered),
        "std_us": statistics.pstdev(ordered),
        "min_us": ordered[0],
        "p10_us": _percentile(ordered, 0.10),
        "p90_us": _percentile(ordered, 0.90),
    }


def _candidate_src_id() -> str:
    try:
        import hashlib

        cu = KERNEL_DIR / "src" / "csrc" / "qknorm_rope_kernel.cu"
        h = hashlib.sha1(cu.read_bytes()).hexdigest()[:12]
        return f"{cu.name}@{h}"
    except Exception as exc:  # noqa: BLE001
        return f"unknown({exc!r})"


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
    mod = _load_correctness_module()
    cases = [c for c in mod.make_cases() if c["kind"] == "production"]

    host = socket.gethostname()
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    gpu_name = torch.cuda.get_device_name(0)
    try:
        import sglang

        sglang_version = getattr(sglang, "__version__", "unknown")
    except Exception:
        sglang_version = "unknown"
    kp_commit = os.environ.get("BENCH_KP_COMMIT", "unset")
    command = "python " + " ".join(sys.argv)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    candidate_src = _candidate_src_id()
    util_before, mem_before = _gpu_state()  # GPU idleness before any timed work

    # Warm the CUDA extension once, and bind both callables once (binding inside
    # the timed loop would charge per-call Python import/exec overhead).
    sys.path.insert(0, str(KERNEL_DIR / "src"))
    import wrapper  # noqa

    wrapper.build()
    from sglang.jit_kernel.diffusion.qknorm_rope import fused_inplace_qknorm_rope as baseline_fn

    candidate_fn = wrapper.fused_inplace_qknorm_rope

    rows: list[dict[str, Any]] = []
    speedups: dict[str, list[float]] = {"all": [], "large": [], "tiny": []}
    print(f"# host={host} gpu={gpu_id}:{gpu_name} sglang={sglang_version} kp_commit={kp_commit}")
    print(f"# {'case':22s} {'bucket':6s} {'baseline_us':>12s} {'cand_us':>10s} {'speedup':>8s} {'cand_path':>9s}")

    for case in cases:
        inp = mod.build_inputs(case)
        q0, k0 = inp["q"].clone(), inp["k"].clone()
        qb, kb = q0.clone(), k0.clone()  # work buffers
        qw, kw = inp["q_weight"], inp["k_weight"]
        csc, pos = inp["cos_sin_cache"], inp["positions"]
        is_neox, eps, rope_dim = inp["is_neox"], inp["eps"], inp["rope_dim"]

        def reset() -> None:
            qb.copy_(q0)
            kb.copy_(k0)

        def run_baseline() -> None:
            baseline_fn(qb, kb, qw, kw, csc, pos, is_neox=is_neox, eps=eps, rope_dim=rope_dim)

        def run_candidate() -> None:
            candidate_fn(qb, kb, qw, kw, csc, pos, is_neox=is_neox, eps=eps, rope_dim=rope_dim)

        base_stats = _time_impl(run_baseline, reset)
        reset()
        run_candidate()
        cand_path = None
        try:
            cand_path = wrapper.last_dispatch_path()
        except Exception:
            pass
        cand_stats = _time_impl(run_candidate, reset)

        speedup = base_stats["median_us"] / cand_stats["median_us"]
        speedups["all"].append(speedup)
        speedups[case["bucket"]].append(speedup)

        for impl, stats, path, sp in (
            ("sglang_baseline", base_stats, "baseline", ""),
            ("candidate", cand_stats, cand_path, f"{speedup:.4f}"),
        ):
            rows.append(
                {
                    "timestamp": ts, "host": host, "gpu_id": gpu_id, "gpu_name": gpu_name,
                    "case": case["name"], "bucket": case["bucket"],
                    "num_tokens": case["num_tokens"], "num_heads": case["num_heads"],
                    "head_dim": case["head_dim"], "rope_dim": case["rope_dim"],
                    "is_neox": case["is_neox"], "eps": case["eps"], "impl": impl,
                    **{kf: round(stats[kf], 4) for kf in
                       ("median_us", "mean_us", "std_us", "min_us", "p10_us", "p90_us")},
                    "dispatch_path": path, "speedup_vs_baseline": sp,
                    "sglang_version": sglang_version, "candidate_src": candidate_src,
                    "kp_commit": kp_commit, "gpu_util_before": util_before,
                    "gpu_mem_before_mb": mem_before, "command": command,
                }
            )
        print(f"  {case['name']:22s} {case['bucket']:6s} {base_stats['median_us']:12.3f} "
              f"{cand_stats['median_us']:10.3f} {speedup:8.4f} {str(cand_path):>9s}")

    def geomean(xs: list[float]) -> float:
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

    geo_all = geomean(speedups["all"])
    geo_large = geomean(speedups["large"])
    geo_tiny = geomean(speedups["tiny"])
    # Settle to a TRUE after-idle snapshot: drop the work buffers + allocator
    # cache, then poll until GPU utilization returns to 0% (the immediate
    # post-timing sample can still catch the just-finished kernel). mem_after then
    # reflects only this process's resident CUDA context (sglang/flashinfer/the
    # JIT extension), recorded as process-owned, not external contention.
    torch.cuda.synchronize()
    try:
        del qb, kb, q0, k0, inp, run_baseline, run_candidate, reset
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    util_after, mem_after = _gpu_state()
    for _ in range(60):  # up to ~30s for nvidia-smi utilization to settle to idle
        if util_after == "0":
            break
        time.sleep(0.5)
        util_after, mem_after = _gpu_state()
    print(f"# GPU before={util_before}%/{mem_before}MB  after-idle={util_after}%"
          f"  (mem_after={mem_after}MB is this process's CUDA context)")
    print(f"# GEOMEAN speedup  all={geo_all:.4f}x  large={geo_large:.4f}x  tiny={geo_tiny:.4f}x")

    idle = {"gpu_util_before": util_before, "gpu_mem_before_mb": mem_before,
            "gpu_util_after": util_after, "gpu_mem_after_mb": mem_after}
    for r in rows:
        r["gpu_util_after"] = util_after
        r["gpu_mem_after_mb"] = mem_after

    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        # GEOMEAN_all written LAST so export.py's last_speedup() reports the
        # all-shape headline speedup (not the tiny bucket).
        for bucket, val in (("tiny", geo_tiny), ("large", geo_large), ("all", geo_all)):
            w.writerow({
                "timestamp": ts, "host": host, "gpu_id": gpu_id, "gpu_name": gpu_name,
                "case": f"GEOMEAN_{bucket}", "bucket": bucket, "impl": "geomean",
                "speedup_vs_baseline": f"{val:.4f}x", "sglang_version": sglang_version,
                "candidate_src": candidate_src, "kp_commit": kp_commit,
                "command": command, **idle,
            })
    print(f"# wrote {len(rows)} rows + 3 geomean rows to {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
