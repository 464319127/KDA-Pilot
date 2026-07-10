"""Adapter stub for the copied standalone benchmark template (bench/benchmark.py).

Documented deviation D3 (docs/benchmark_method.md): this task's kernel is a
world=8 NVLS multicast collective. The template's single-device CUDA-event
timing loop cannot express it — timing one device's event pair around a
spin/flag collective measures the spin of one rank, not the collective, and
the template's per-workload single-`Process` isolation cannot host the
single-process 8-GPU workspace the kernel requires (config.toml
`[benchmark].device`).

The authoritative measurement is bench/ar_harness.py, which preserves every
transferable template policy (frozen workloads.json, fresh inputs per trial,
randomized A/B order, output poisoning, median/mean/std/min/p10/p90, speedup =
baseline_median/candidate_median, no-silent-skip, JSONL provenance records).

This adapter intentionally refuses to run rather than silently measuring the
wrong thing (no-silent-skip applies to wrong measurements too).
"""

_D3_MESSAGE = (
    "This task's kernel is a world=8 NVLS collective; the single-device "
    "template timing loop is structurally inapplicable (documented deviation "
    "D3 in docs/benchmark_method.md). Run the authoritative harness instead: "
    "python3 bench/ar_harness.py --mode {correctness|bench|noise|stability}"
)


def make_case(workload, *, device, seed):
    raise NotImplementedError(_D3_MESSAGE)


def call_baseline(workload, inputs, outputs):
    raise NotImplementedError(_D3_MESSAGE)


def call_candidate(workload, inputs, outputs):
    raise NotImplementedError(_D3_MESSAGE)
