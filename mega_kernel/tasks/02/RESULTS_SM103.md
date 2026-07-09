# 03 — v1 prototype status (B300 devbox, 2026-07-09)

## What was built
`ar_oneshot.cu` + `ar_bench.py`: single-kernel 8-rank one-shot push
AR(+residual add+rmsnorm), CUDA-graph-capturable via device-side epoch
counter (no host flag plumbing — every replay self-advances). Single-process
8-GPU harness with proper `cudaDeviceEnablePeerAccess`, per-device graphs
(50 rounds each) replayed concurrently.

## Results
- **Correctness: 8/8 ranks OK** (out rel 3.1e-3, residual-out rel 3.4e-3 vs
  fp32 oracle). Two v1 bugs found & fixed: norm-phase race (24 x-blocks
  scaling the same rows — gate reduce/norm to blockIdx.x==0) and graph
  capture needing an explicit per-device stream.
- **Perf: 36.6 µs/op at T=6 H=6144 world=8 — 4.2× SLOWER than the
  flashinfer mnnvl fused reference (8.7 µs in serving).**

## Probe matrix (µs/op)

| world | norm | grid_x=8 | 24 | 48 |
|---|---|---:|---:|---:|
| 2 | off | 26.2 | 28.3 | 31.6 |
| 2 | on  | 50.5 | 32.6 | 36.0 |
| 8 | off | 34.7 | 32.2 | 35.4 |
| 8 | on  | 39.0 | 36.5 | 39.9 |

Key finding: cost is nearly world-independent → **~25 µs fixed overhead in
the unicast-push + flag-spin protocol itself**, not wire bandwidth
(73.7 KB × 7 egress = 0.6 µs at NVLink5 rates).

## Root-cause hypotheses (for v2)
1. **The reference kernel's speed comes from NVLS multimem**: flashinfer
   "mnnvl" oneshot uses multicast stores (`multimem.st` — one store fans out
   via NVSwitch) and in-switch reduction (`multimem.ld_reduce`), eliminating
   the 7-way unicast fan-out AND most of the flag protocol. v2 must use
   cuMulticastCreate/BindMem handles — real infra work.
2. Unicast 16B peer stores from one SM likely run far below NVLink BW
   (posted-write pipelining limits); ncu needed with
   `--replay-mode application` (spin-collectives deadlock under kernel
   replay).
3. Norm phase uses 1 block/token (6 blocks total) — latency-bound ~3-5 µs;
   fold into the flag-wait blocks once protocol cost is fixed.

## Verdict
v1 = correct scaffold + honest negative result. Beating 8.7 µs requires the
multimem path; park until that infra is built (or drive flashinfer's own
`trtllm_mnnvl_allreduce` API directly in this harness to A/B and then attack
its residual overhead instead of rebuilding transport from scratch — likely
the smarter route).
