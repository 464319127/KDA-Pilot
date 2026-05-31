# NCU + Roofline REPORT (final) — fused QK-Norm + RoPE on B200 (sm_100)

Authoritative roofline/diagnosis. Host `innomatrix-us-adc-smb200-0003`, GPU 0
NVIDIA B200, verified idle (0% util / 4 MB before; 0% after the profiling run).
Extension JIT-built with `-lineinfo`. Artifacts in this workspace under
`profile/<run>/{harness,reports,analysis}/` for the final candidate and baseline:
- `cand_qwen4096_v3/` — candidate v3, large bucket
- `base_qwen4096/` — SGLang baseline, large bucket
- `cand_qwen19_v3/` — candidate v3, tiny bucket
Each `reports/` has both `full.ncu-rep` (`ncu --set full`) and `source.ncu-rep`
(`ncu --set source --section SourceCounters`); `harness/` holds the exact
`profile_entry.py` + `profile_cmd.sh`; `analysis/` holds `sol_summary.txt`,
`full_metrics.csv`, `source_counters.txt`.

## Active bound: LATENCY-bound, not DRAM-bandwidth-bound

Cold-cache roofline (`roofline_cold.py`; pristine Q/K reset + L2 flush before each
sample; bytes = `4·N·H·D·2 + N·rope·4`):

| shape | impl | cold µs | BW TB/s | % of ~8 TB/s peak |
|---|---|---|---|---|
| qwen__4096 | baseline | 43.3 | 2.37 | 29.7 |
| qwen__4096 | **cand v3** | **36.1** | **2.85** | **35.6** |
| joyai__7904 | baseline | 91.0 | 2.89 | 36.1 |
| joyai__7904 | **cand v3** | **73.1** | **3.60** | **45.0** |
| qwen_edit__8424 | baseline | 74.8 | 2.83 | 35.3 |
| qwen_edit__8424 | **cand v3** | **60.8** | **3.48** | **43.5** |
| zimage__4128 | baseline | 50.8 | 2.54 | 31.7 |
| zimage__4128 | **cand v3** | **41.6** | **3.10** | **38.7** |

Both kernels reach only ~30–45% of HBM peak **even cold** → the limiter is
latency (small per-(token,head) in-place read-modify-write, a warp/half-warp
reduction dependency, and limited memory-level parallelism), not bandwidth.

## NCU Speed-of-Light — large bucket `qwen__4096` (warm/L2-resident, `--set full`)

| metric | baseline | cand v3 |
|---|---|---|
| Duration (NCU, µs) | 58.6 | **49.8** |
| Compute (SM) SOL % | 58.1 | 58.7 |
| Memory SOL % | 48.4 | 34.8 |
| DRAM throughput % | 13.6 | 16.0 |
| Achieved occupancy % | 88.0 | 76.8 |
| Registers/thread | 32 | 32 |
| Uncoalesced-access rule | **28% excess, Est 23.9%** | not flagged |

Candidate kernel GPU time is ~15% below the baseline (49.8 vs 58.6 µs), matching
the wall-clock win. Candidate Memory SOL is *lower* because the coalesced cos/sin
load (v2) removed the wasted sectors the baseline still incurs. Source-counter
report (`reports/source.ncu-rep`, parsed `analysis/source_counters.txt`) confirms
no remaining strided cos/sin gather on the candidate.

## NCU — tiny bucket `qwen__19` (candidate v3, `--set full`), measured classification

| metric | value | reading |
|---|---|---|
| Duration (NCU, µs) | 7.8 | kernel GPU time |
| Waves Per SM | **0.05** | the 19-token workload cannot fill 148 SMs |
| Compute (SM) SOL % | 1.8 | GPU ~98% idle during the kernel |
| Memory SOL % / DRAM % | 2.4 / 0.5 | not memory-bound |

Measured evidence that the tiny bucket is **launch / occupancy-bound**: the
problem is too small to fill the GPU (≈0.05 waves/SM, <2% SOL), so wall-clock
latency is dominated by launch/dispatch, not kernel work. NCU's occupancy rule
(Est. 52%) reflects the inherent under-fill of a 19-token problem, which no kernel
change can remove — consistent with treating the tiny bucket as launch-bound and
not chasing kernel-level speedups there. (This is the measured basis for "skip
further tiny-shape kernel optimization", per the profiling golden rule.)

## Diagnosis → design changes (KDA loop)

1. **v1→v2**: NCU flagged a strided/uncoalesced cos/sin gather (28% excess
   sectors). Fixed with a coalesced `float2 __ldg` per lane. (Baseline still
   carries this pattern — see its `Est 23.9%` above.)
2. **v2→v3** (the win): cold-cache roofline showed both kernels at ~30% of peak →
   latency/low-MLP bound. Switched to **2-heads-per-warp + 128-bit (float4)**
   load/store (each lane owns 8 bf16, 16 lanes/head, half-warp RMS reduction,
   cos/sin float4 per lane). More bytes-in-flight per warp hides the latency:
   cold BW ~30%→~40% of peak, NCU duration 58.6(base)/60→**49.8** µs.

## Per-bucket result (final per-call benchmark, pristine reset per sample)

- **Large (4096–8424)**: latency-bound (cold ~36–45% of peak). v3 geomean
  **1.133×** over baseline (joyai 1.212×, qwen 1.100×, qwen_edit 1.158×, zimage
  1.101×/1.096×).
- **Tiny (19–195)**: launch/occupancy-bound (0.05 waves/SM, <2% SOL). v3 geomean
  **1.091×** (native dispatch lighter than the tvm-ffi baseline). Tiny per-call
  latency (~24–26 µs) is dominated by launch/dispatch latency from an idle GPU,
  not kernel compute (~7.8 µs per NCU); the per-sample reset + `synchronize()`
  happen before `start.record()` and are excluded from the CUDA-event window.
- **All 10**: geomean **1.111×**. Correct 21/21. (Single-call timing carries
  ~±0.02 run-to-run variance on the geomeans; the candidate wins every shape
  across runs — see `benchmark.csv`.)

## Near-bound judgment

The candidate is ~40% of HBM peak (cold) and clearly ahead of the well-tuned
baseline on every shape. The remaining gap to peak is structural (in-place RMW +
tiny per-token work + reduction dependency); the one untried lever (multiple
tokens/thread for ILP) is not expected to yield a large further gain (Codex
task8 concurs). Near the attainable bound for this in-place kernel structure,
with a clear, measured, evidence-backed win → promoted.
