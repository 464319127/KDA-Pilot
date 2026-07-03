# KDA Prompt: glm52_bs1__mtp_chain_megakernel

Target GPU: NVIDIA B300 (sm_103). Fuse the **6-step MTP draft chain** (M=1,
sequential, fully shape-static) into as few kernels as possible — ideally one
persistent megakernel — **~11% of the round** (~2.0 ms including the extend).
This is the *draft side*: numerics changes are accept-SAFE (a more accurate
MoE kernel here raised accept 5.0→5.2), so it is the free-swing megakernel
playground. TileRT's whole-model megakernel approach applies 1:1 to this
single-layer chain.

## Problem (FIXED)

Per chain step i (i = 1..6), batch of exactly 1 token:

```
d_i   = argmax(logits_{i-1})                       [1] int32
x     = eh-less MTP step:  h = W_eh @ cat(RMS(emb(d_i)), RMS(hidden_{i-1}))
        -> 1x GLM decoder layer (MLA attn @pos p_i over its own KV slot,
           page 64, kv_len = p_i+1; MoE: 256 experts topk 8 + shared)
        -> shared_head_norm -> logits_i = lm_head(h_i)   [1, 18944 shard]
KV    : chain step writes MTP-layer KV at position p_i (combined ckv|kpe pool)
output: 6 draft token ids + the 6 hidden states stay internal
```

Weights: the single MTP layer (fp8 block-scale experts, bf16 dense) + shared
lm_head shard + embedding — all static, ~2.4 GB/rank; TP=8 with 2
all-reduces per step (attention o_proj, MoE out) and one lm_head
all-gather/argmax (vocab-sharded: local argmax + cross-rank max-reduce).

## Baseline (measured)

Fused chain CUDA graph in mini-sglang (`_capture_mtp_chain`, commit
`a26fd6f`): 6 steps ≈ **1.3-1.5 ms** (per step ~220 µs: layer ~180 µs + head
+ argmax + scatter). The per-step kernel count is ~40; most are < 5 µs —
launch gaps and tail effects dominate, which is exactly what a
persistent/PDL-chained megakernel removes.

**Success: 6-step chain ≤ 0.8 ms end-to-end** (same drafts as baseline on
greedy: token-exact match on ≥ 99% of steps over a 1k-round replay corpus;
bitwise hidden match NOT required — draft side).

## Approach notes

- Warp-specialized persistent kernel: dedicate warp groups to
  GEMM-tile / attention / MoE-expert roles; the M=1 GEMMs are pure GEMV
  (memory-bound) so tensor cores optional except lm_head.
- Cross-rank: the 2 all-reduces per step can use the NVLS one-shot from
  `glm52_bs1__oneshot_allreduce_bf16` (compose tasks) or multimem inline in
  the megakernel; grid-wide sync via cooperative groups or PDL chain of 2-3
  kernels per step is acceptable — "one kernel" is not the goal, ≤ 0.8 ms is.
- The argmax feeds the next step's embedding lookup in-kernel (prior art does
  this in-graph already; keep it on-GPU).
- KV append is 576 bf16 per step — trivial store, keep it in the kernel.

## Hardware access (B300)

Full runbook: `../docs/b300_access.md`. **Whole 8-GPU node required** (TP=8
weights + 2 all-reduces per chain step):

```bash
export PATH="$HOME/.local/bin:$PATH"; export RADIX_API=https://nodes.sglang.io
radix assign verda-b300-fin-03-3       # free re-assign on lease lapse (4h, cannot extend)
ssh -i ~/.ssh/id_ed25519 -J ubuntu@95.133.252.66 bbuf@light-face-hides-fin-03-3
nvidia-smi                              # all 8 free? (glm_pd tenant; fall back to fin-03-2/-4)
docker exec -it sglang_new bash        # weights /data/bbuf/glm52_real, repo /data/bbuf/repos/mini-sglang
```

The baseline chain graph only exists inside a live mini-sglang server — for
isolated timing, replay the captured chain via the harness you build in
`bench/` (extract the MTP layer weights once from the rank shards). Launch
servers detached (`docker exec -d`, log to /data/bbuf), never via ssh `&`.

Follow `../../llm/docs/llm_kernel_optimization_rules.md` +
`../../llm/docs/llm_correctness_contract.md`. Baseline source = mini-sglang
`python/minisgl/engine/graph.py::_capture_mtp_chain` + the layer modules it
replays (`models/glm_moe_dsa.py`); copy into `baseline/` with SHAs.
