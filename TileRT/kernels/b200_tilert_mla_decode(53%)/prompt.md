# b200_tilert_mla_decode (53%)
Target GPU: NVIDIA B200 (sm_100).
## Problem — THE #1 decode cost (52.8% of TileRT decode CUDA time)
MLA (Multi-head Latent Attention) decode, matching TileRT `PureMlaDsv32`. DeepSeek-V3.2
uses DSA sparse attention: GPU0 indexer selects top-2048 KV; workers attend over the
selected KV. For context ≤ index_topk=2048 (e.g. GSM8K) this == dense MLA.
Math (absorbed-free, correct): with compressed KV latent cache,
```
kv = kv_c @ Wkv_b -> [n_local_heads, qk_nope + v_head]; k_nope, v
q  = [q_nope, q_pe(rope)] ; k = [k_nope, k_pe(rope shared)]
o  = softmax(q·k^T / sqrt(192)) · v ; out = o @ Wo
```
Shapes (decode, per worker, n_local_heads=20 padded): q_nope[1,20,128]+q_pe[1,20,64] bf16;
kv latent cache top-2048: kv[.,2048,512]+pe[.,2048,64] bf16; indices[1,2048] i32 -> o[1,20,512] bf16.
## TileRT reference
PureMlaDsv32 = **35.2µs/call in-graph (52.8% of decode)** but **11.68µs isolated**
(≥3× ncu median, `gpu__time_duration.avg`, HBM 2.7% → launch/latency-bound, not
BW-bound; s2 12.45 / s4 12.48µs). 168 reg, occupancy=1. **The KDA target is the
isolated number** (`config.toml [reference]` = 11.68µs); the candidate is measured the
same isolated way.
**Open-source check (NOT beaten):** flashinfer BatchMLAPagedAttentionWrapper on
kv_len=2048 = **25.5µs isolated** (rel 2.6e-3 vs torch) — that is **~2× SLOWER than
TileRT's 11.68µs**, not faster. (An earlier note here read "flashinfer can beat
TileRT" — that compared flashinfer-isolated 25.5µs against TileRT *in-graph* 35.2µs,
which is apples-to-oranges. Fair isolated-vs-isolated: TileRT wins.) So this op still
needs a faster custom kernel; there is no free open-source win.
## Design levers to exploit (see ../../docs/tilert_design_levers.md)
- **L1 tile overlap** (§4/§13): persistent CTA (168 reg, occupancy=1), Prefetcher TMA-streams
  the gathered KV tiles while Consumer warps run warpgroup HMMA; mbarrier double-buffer.
- **L2 no-GMEM intermediates** (§13.3): keep q·kᵀ scores + softmax state in SMEM/TMEM.
- **L5 DSA bandwidth** (§16/§20): read only the **top-2048** rows of the compressed latent
  cache (kv_c 512 + pe 64) — KV-read scales with 2048, not seq_len. This is the #1 decode op.

## Shapes (decode + MTP)
PureMlaDsv32 SASS kSeqLen = {1,2,4} (decode=1, MTP verify=4; seq=3 lives in the
`flash_sparse_mla` task). Workloads: `bench/workloads.json` (S1/S2/S4 at kv_len=2048).
Validated by `../../harness/tilert_oracle.py case_flash_sparse_mla` (rel ~3.0e-3 vs real op).

## Goal
CUDA MLA-decode kernel matching the baseline output and **≤ ~11.7µs isolated** at
kv_len 2048, on every S∈{1,2,4}. The bar is TileRT's **11.68µs ≥3× ncu median**
(`config.toml [reference]`) — beating flashinfer's 25.5µs is necessary but NOT
sufficient. Lever: warp-spec persistent + sparse KV gather.
