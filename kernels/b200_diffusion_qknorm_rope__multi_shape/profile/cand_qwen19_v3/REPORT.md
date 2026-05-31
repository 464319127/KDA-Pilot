# NCU REPORT — candidate v3, tiny shape `qwen__19` (B200, sm_100)

Run dir: `profile/cand_qwen19_v3/`. Host `innomatrix-us-adc-smb200-0003`, GPU 0
NVIDIA B200, verified idle. Captured with `ncu --set full --profile-from-start
off` and `ncu --set source --section SourceCounters`. Artifacts:
`reports/{full,source}.ncu-rep`, `harness/{profile_entry.py,profile_cmd.sh}`,
`analysis/{sol_summary.txt,full_metrics.csv,source_counters.txt}`.

Shape: `q=k=[19, 24, 128]` bf16, `rope_dim=128`, `is_neox=False`. This run gives
the measured basis for classifying the **tiny bucket**.

## Speed-of-Light (from `analysis/sol_summary.txt`)

| metric | value |
|---|---|
| Duration | 7.84 µs |
| Waves Per SM | **0.05** |
| Compute (SM) Throughput | 1.81 % |
| Memory Throughput | 2.41 % |
| DRAM Throughput | 0.49 % (32 GB/s) |
| Registers / thread | 32 |

## Active limiter

**Launch / occupancy-bound — the problem is too small to fill the GPU.** A
19-token workload is `19 × (24+24) = 912` head-vectors → ~456 warps with the
2-heads-per-warp mapping, i.e. **≈0.05 waves across 148 SMs**: the GPU is ~98%
idle during the kernel (Compute SOL 1.8%, DRAM 0.5%). The per-call wall-clock
latency for the tiny shapes (benchmark.csv: ~25.6–26.0 µs baseline, ~23.5–23.8 µs
candidate) is therefore dominated by launch/dispatch latency from an idle GPU,
not kernel compute: the kernel itself is only ~7.8 µs (this NCU duration), and the
per-sample reset + `synchronize()` are excluded from the CUDA-event window (they
run before `start.record()`). The latency is also roughly flat across tiny token
counts (19 vs 195), the signature of launch-bound work. NCU's occupancy rule
(Est. ~52%) reflects this inherent under-fill; no kernel change can add work that
isn't there, so further tiny-shape kernel optimization is not pursued (per the
profiling golden rule: a profile would not change the next edit). The candidate's
tiny-bucket win (geomean 1.0905×) comes from a lighter native dispatch path than
the tvm-ffi baseline, not from kernel-level compute.
