# NCU REPORT — SGLang baseline `fused_qknorm_rope_warp`, shape `qwen__4096` (B200, sm_100)

Run dir: `profile/base_qwen4096/`. Host `innomatrix-us-adc-smb200-0003`, GPU 0
NVIDIA B200, verified idle (0%/0 MB before & after the profiling pass). Captured
with `ncu --set full --profile-from-start off` (one bracketed launch) and
`ncu --set source --section SourceCounters`. Artifacts:
`reports/{full,source}.ncu-rep`, `harness/{profile_entry.py,profile_cmd.sh}`,
`analysis/{sol_summary.txt,full_metrics.csv,source_counters.txt}`.

This is the reference baseline for the candidate; metrics let the candidate's win
be read against an apples-to-apples profile of the kernel under optimization.

## Speed-of-Light (from `analysis/sol_summary.txt`)

| metric | value |
|---|---|
| Duration | 58.62 µs |
| Compute (SM) Throughput | 58.06 % |
| Memory Throughput | 48.40 % |
| DRAM Throughput | 13.61 % (904 GB/s) |
| Achieved Occupancy | 88.00 % |
| Registers / thread | 32 |
| Waves Per SM | 1 |

## Active limiter

**Latency-bound, not DRAM-bandwidth-bound.** DRAM throughput is only 13.6%; no
single SOL is saturated (SM 58%, memory 48%). The kernel is a small per-(token,
head) in-place read-modify-write with a warp reduction — limited memory-level
parallelism, so latency dominates. The NCU rule engine flags **uncoalesced global
accesses, Est. Speedup 23.87%** (the strided scalar cos/sin gather), confirmed in
`analysis/source_counters.txt`; this is the inefficiency the candidate removed
(coalesced `float2` cos/sin) and then improved on with 2-heads-per-warp + 128-bit
loads. Candidate v3 runs the same shape in 49.8 µs (≈15% faster) — see
`../cand_qwen4096_v3/REPORT.md`.
