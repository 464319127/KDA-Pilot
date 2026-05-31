# NCU Report — fused QK-Norm + RoPE, shape `qwen__4096` (B200, sm_100)

Shape: `q=k=[4096, 24, 128]` bf16, `rope_dim=128`, `is_neox=False`, `eps=1e-6`.
Host `innomatrix-us-adc-smb200-0003`, GPU 0 NVIDIA B200 (verified idle: 0% util, 0 MB before).
Captured with `ncu --set full --profile-from-start off` on one bracketed launch
(warmup + input reset excluded). Extension built with `-lineinfo`.
Raw reports (REMOTE_KDA_DIR): `profile/{cand_qwen4096_v1,cand_qwen4096_v2,base_qwen4096}/reports/full.ncu-rep`.

## Speed-of-Light comparison

| metric | cand v1 | cand v2 | SGLang baseline |
|---|---|---|---|
| Duration (NCU, us) | 62.62 | 60.13 | 59.04 |
| Compute (SM) SOL % | 58.32 | ~ (see note) | ~ |
| Memory SOL % | 43.47 | 36.81 | 47.53 |
| DRAM throughput % | 12.73 | 13.28 | 13.52 |
| DRAM GB/s | ~845 | ~882 | ~898 |
| Achieved occupancy % | 86.95 | 88.15 | 89.16 |
| Registers / thread | 28 | 28 | 32 |
| Grid / Waves-per-SM | 1184 / 1 | 1184 / 1 | 1184 / 1 |
| Uncoalesced flag | 28% excess, Est 24.66% | removed | 28% excess, Est 23.75% |

Wall-clock benchmark (CUDA events, no NCU overhead): baseline 30.94 us, cand v1
34.91 us (0.886x), cand v2 32.94 us (0.939x).

## Six analysis dimensions

1. **Occupancy** — Not the bottleneck. Theoretical 100%, achieved ~88%; 28
   regs/thread, block limit = warps (8/SM), 1 wave. Healthy.
2. **Balance (compute vs memory)** — DRAM throughput is only ~13% on all three
   kernels: the 50 MB q + 50 MB k working set is **L2-resident** under repeated
   launches, so this is **not DRAM-bandwidth-bound**. The limiter is L2 / issue /
   latency. cand v2 Memory SOL (36.8%) is *lower* than baseline (47.5%) because
   the coalescing fix removed wasted sectors (less work, same result).
3. **Stalls / latency** — NCU notes "typically indicate latency issues"; with 1
   wave and a short kernel, per-warp latency hiding is the practical limiter.
4. **Tensor core** — N/A (no MMA); correctly absent (over-engineering avoided).
5. **Timeline / tail** — single wave, uniform work per (token,head); no tail
   imbalance (fixed-length head vectors).
6. **Memory pattern** — v1's cos/sin gather used strided scalar `__ldg`
   (`cos[2L]`, `cos[2L+1]` per lane → 8-byte stride → 2x sector over-fetch):
   **28% excessive sectors, Est. 24.66%**. The SGLang baseline carries the same
   pattern (Est. 23.75%). v2 loads cos/sin as one coalesced `float2 __ldg` per
   lane → the uncoalesced flag is **removed** for the candidate.

## Diagnosis (playbook match)

Memory-access-pattern inefficiency (uncoalesced gather) on v1 → fixed in v2 by
vectorizing the cos/sin read (the only consumer with a strided pattern). After
the fix the candidate kernel is at **near parity** with the hand-tuned SGLang
baseline in GPU time (60.1 vs 59.0 us) and ahead on the launch-bound tiny shapes.
Neither kernel is near the DRAM roofline under this L2-resident measurement.

## Resulting design change & next directions

- Applied (v1→v2): coalesced `float2` cos/sin load + FMA reduction →
  geomean large 0.889x→0.927x, tiny 1.145x→1.187x, all 1.009x→1.049x.
- Next (later round): 128-bit (16-byte) vectorized load/store via 2-heads-per-warp
  to cut issue count on large shapes (residual NCU Est. ~10%); and a cold-cache
  (L2-flush) benchmark variant to report the true DRAM-bound roofline alongside
  the L2-resident numbers.
