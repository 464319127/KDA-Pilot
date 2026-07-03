# Profile evidence — glm52_bs1__mla_decode_bs_consistent

MLA decode ~17 µs/layer @B=8 x 78 layers ≈ 1.4 ms/round (~8%) via
flashinfer trtllm_batch_decode_with_kv_cache_mla (sm_100f cubin on sm_103;
needs dedicated 512MB workspace; page {32,64}; block_num must be even).

Consistency evidence (why bs-invariance is a deliverable): with k=7
semantics unchanged, replaying verify+extend on the bs-9 graph collapsed
accept 5.16 -> 3.19 while GSM8K held 91.7% — per-batch-size kernel numerics
are the mechanism gating k<=7 today. Full experiment log: mini-sglang
handoff doc (prompt.md, 2026-07-03 sections) + commit a26fd6f message.
