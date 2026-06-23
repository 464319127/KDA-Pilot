# Benchmark Method & Frozen Baseline — fast_topk_transform_fused (B200)

## Harness
- `bench/benchmark.py` is a byte-identical copy of `llm/docs/standalone_llm_benchmark_template.py`
  (no bespoke harness). CUDA-event timing, inner-loop amplification to ~1000us, interleaved A/B
  per trial, isolated subprocess per workload, warmup=10, num_trials=7, outputs preallocated and
  excluded from the timed region; per-workload median/mean/std/min/p10/p90 + per-workload speedup
  (`baseline_median/candidate_median`) + equal-weight geomean over production rows.
- `bench/adapter.py` make_case/call_baseline/call_candidate + a regime-aware per-workload
  `compare_outputs` sanity check (naive exact; radix order-tolerant sorted-set). The AUTHORITATIVE
  correctness gate is `bench/correctness.py` (`matched_ratio==1.0`, R3/R4), run before benchmarking.

## Environment / provenance
- Host: `ion-b200` → `innomatrix-us-adc-smb200-0003`; container `sglang_bbuf`; workspace
  `/home/sglang-omni/bbuf/kda_topk`.
- Toolchain: torch `2.11.0+cu130`, CUDA `13.0`, nvcc `13.0.88`, `tvm_ffi 0.1.9`.
- Pinned GPU: id 1 (`CUDA_VISIBLE_DEVICES=1`, `--device cuda:0`).
- Baseline source: SGLang `main` commit `7e6587c94a1d0305815a14067c5d3cc02a9b0f36` (see baseline_source.md).
- Candidate source hash (current = baseline-forwarding stub): `candidate_topk_transform.cu`
  sha256 `a6e7835a486bc3b1db7914b65ff0ac5e52824d732a37fb1557e9b097d2af823a`; `binding.cu` sha256
  `d8058ec2bfcc06306c012df7b0aec3d74a2f8071d5412aa8065708e518f116ff`.
- ABI build flags (one module, symmetric; `solution/build.py`): `nvcc -O3 -std=c++17 -lineinfo
  -gencode=arch=compute_100,code=sm_100 -D_GLIBCXX_USE_CXX11_ABI=<torch>`; **no fast-math**; torch
  include/lib linked (`-ltorch -ltorch_cpu -lc10 -ltorch_cuda -lc10_cuda`).

## Command
```
CUDA_VISIBLE_DEVICES=1 python3 bench/benchmark.py --device cuda:0 \
  --workloads bench/workloads.json --out bench/results.jsonl
# template defaults: --warmup-runs 10 --num-trials 7 --inner-iterations-min 1
#   --inner-iterations-max 4096 --target-sample-us 1000 --timeout-seconds 600 (isolated)
```

## GPU-1 idle evidence (AC-7)
- Before: GPU 1 `util 0%`, `59666 MiB` used (a parked process pid 3048677 sharded GPU0/1, **util 0%**,
  no active compute). No cleanly-idle GPU existed (3/4/5/6/7 were 149-157 GB used or util 7-73%).
- After: GPU 1 `util 0%`, `59666 MiB` used — unchanged; the parked process never woke, so no compute
  interference during the run.
- Decision: `util 0%` = no active-compute contention → timing valid on the pinned GPU. The 59 GB
  resident parked allocation does not contend for SMs/bandwidth; documented as a caveat.

## Frozen immutable baseline (candidate == baseline stub this run; speedup ≈ 1.0 confirms a fair harness)
- 251 workloads: **250 PASSED, 1 INCORRECT** (`reg_ties_boundary`, a radix+exact-ties REGRESSION row;
  the order-tolerant `compare_outputs` cannot resolve tie-set ambiguity without inputs — it is
  `production=false` and is authoritatively validated by `correctness.py` valid-top-k; not in the headline).
- **Production headline (236 rows): geomean_speedup 0.9977, arithmetic 0.9979, min 0.80, max 1.097.**
  Baseline-vs-itself ≈ 1.0 confirms the harness is fair; the per-row spread (~±20%) is the
  **tiny-kernel measurement noise floor** — a real candidate win must robustly clear it.
- Per-regime baseline `median_us` (production):
  - decode (naive, `S==B`, dominant): n=212, p50 **9.32 µs** (min 7.63, max 13.59).
  - radix (`length>topk`, large-B prefill): n=24, p50 **9.29 µs** (min 8.22, **max 74.79 µs**).
- Raw per-run `samples_us` arrays trimmed from the committed `bench/results.jsonl` (stats retained).

## Notes
- These baseline numbers are **immutable**: a candidate is benchmarked by re-running this same command
  (which re-times both sides in one fair run); the workloads/settings are frozen and not changed.
- Active-bound to confirm with NCU once the native candidate exists (per `docs/dispatch.md`): decode
  ≈ store/launch-overhead bound at ~9 µs; large-B prefill ≈ selection/store bound at ~75 µs.
