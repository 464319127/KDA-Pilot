# b200_tilert_mla_decode
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
PureMlaDsv32 = **35.2µs/call avg, 52.8% of decode** (profiler). 168 reg, occupancy=1.
Reference point: flashinfer BatchMLAPagedAttentionWrapper on kv_len=2048 = **25.5µs**
(rel 2.6e-3 vs torch) — i.e. a well-tuned MLA decode can beat TileRT here.
## Goal
CUDA MLA-decode kernel matching the baseline output and ≤ ~35µs at kv_len 2048
(stretch: ~25µs like flashinfer). Lever: warp-spec persistent + sparse KV gather.
