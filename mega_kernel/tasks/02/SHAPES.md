# 03_ar_norm_fused_bs1 — 8-rank oneshot allreduce(+add+rmsnorm), bs=1

Payloads (bf16, hidden=6144, TP=8, NVLink NV18 full mesh, single node):

| tokens | bytes | calls/iter | site |
|---:|---:|---:|---|
| 6 | 73,728 B | ~156 | verify layers (attn-out AR+norm, MoE-out AR+norm, fused) |
| 1 | 12,288 B | ~8-10 | draft steps (unfused, `one_shot_push`) |
| 6 | 73,728 B | ~2-4 | draft-extend / last-layer (unfused) |

## Baseline
flashinfer `trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel`: **8.7 µs**
in-graph mean (fused AR+residual+rmsnorm). Unfused sgl `one_shot_push`:
median 8 µs (mean inflated by rank-arrival skew — median is the honest
kernel number).

## Analysis
73 KB over NVLink5 peer writes: wire floor < 1 µs; observed 8.7 µs is
latency/handshake dominated (flag exchange, block sync, grid ramp).
TileRT's flag-based AR fused into GEMM epilogues implies ~2-4 µs is
achievable per op at this size.

## Target
≤5 µs fused AR+add+rmsnorm at 6-token payload (−0.6 ms/iter at 160 calls);
stretch: fuse into the preceding GEMM epilogue (o_proj / MoE finalize) like
TileRT `UnprojOAllreduce` / `ExpertDownAllreduce`.

## Design notes
- One-shot push over cudaMemPool-exchanged peer buffers + arrival flags
  (all 8 ranks write their shard to all peers, spin on 7 flags, reduce
  locally, then add+rmsnorm in the same kernel).
- Must be CUDA-graph-capturable: flag values must rotate per replay from a
  graph-owned counter (TileRT passes `flag` as a kernel arg baked per node —
  we need an in-graph atomically-bumped epoch instead).
- Correctness gate: bitwise vs current fused path on fixed inputs across
  1000 replays (flag-reuse races show up as flaky mismatches).
- NCU + inter-rank timestamp deltas (clock64 stamps) for skew accounting.

## Serving hook
`flashinfer_comm_fusion.py` / `communicator.py` allreduce-fusion dispatch —
add backend option alongside `mnnvl`/`trtllm`.
