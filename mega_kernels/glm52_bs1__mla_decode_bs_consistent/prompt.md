# KDA Prompt: glm52_bs1__mla_decode_bs_consistent

Target GPU: NVIDIA B300 (sm_103). MLA absorbed decode attention for GLM-5.2
bs=1 spec decode — **~8% of the round** (~1.4 ms), with a second prize that
is potentially bigger than the speedup: **batch-size-consistent numerics**.

## Problem (FIXED)

Absorbed-MLA paged decode (DeepSeek-V3.2 geometry), per layer x78:

```
q      [B, 16, 576] bf16   (nope 512 | rope 64, heads-per-rank 16)
kv     paged pool [pages, 64, 576] bf16 (combined ckv|kpe, page 64)
out    [B, 16, 512] bf16, softmax scale 1/sqrt(576)
B      ∈ {1..8} (chain=1, verify=8), kv_len per row ∈ [1k..8k] typical,
       rows of one verify batch are STAGGERED: kv_len_i = L+i
block_tables [B, W] int32 (page ids), W even (trtllm constraint today)
```

## Baseline (measured)

flashinfer `trtllm_batch_decode_with_kv_cache_mla` (sm_100f cubin runs on
sm_103): ~17 µs/layer at B=8, needs a dedicated 512 MB workspace, page ∈
{32,64} only. It is the fastest known baseline — beating it by ≥10% at B=8
is the speed goal (~15 µs), B=1 proportionally.

## The consistency prize (why this task exists)

Measured on the e2e system: replaying verify on the bs-9 graph instead of
bs-8 — with IDENTICAL round semantics — collapses MTP accept 5.16 → 3.19
(GSM8K unharmed). Per-batch-size kernel scheduling (this cubin's bs-dependent
split, nvjet's M-dispatch) changes the model's numerics style, and the deep
draft chain cannot survive the mismatch between the bs it was "calibrated"
on and a new bs. Today this pins the deployment at verify batch = 8 and
forbids k > 7.

**A candidate whose per-row reduction order is IDENTICAL for every B ∈ 1..10
(row-independent scheduling: split by kv chunks per row, never by batch
slot count) removes the attention term from that mismatch.** Combined with
the already-M-invariant custom MoE + M-stable dense GEMMs, it opens k=8-9
(accept ~5.8-6.2, +8-10% e2e). Deliver `consistency_report.md`: bitwise
row-output equality for the same (q, kv_len) across B = 1, 8, 9, 10.

**Success (either counts, both is the jackpot):**
1. ≥10% faster than the trtllm cubin at B=8, rel err ≤ 1e-3 vs fp32 oracle, or
2. within 15% of the cubin AND bitwise B-consistent per row.

## Approach notes

- Head dim 576/512 with 16 heads: one CTA per (row, kv-chunk), fp8 KV is NOT
  in scope (pool is bf16). Online-softmax with fp32 running max/denominator;
  fixed split-size (e.g. 1024 tokens per chunk) + a deterministic tree
  combine gives B-invariance for free — variable split-count-by-total-work is
  what breaks it in the cubin.
- The staggered kv_len pattern means row i and row i+1 share all but one page
  — L2 reuse across rows is nearly total; don't fight it with row-parallel
  rasterization that spreads rows across SM partitions.
- Prior integration points: mini-sglang `python/minisgl/attention/mla.py`
  (`_bind_capture_buffers`, lean decode metadata) — the candidate slots in
  behind `MINISGL_MLA_BACKEND`.

## Hardware access (B300)

Full runbook: `../docs/b300_access.md`. Kernel loop fits on ONE idle GPU
(attention is per-rank; the heads-per-rank geometry is already in the shapes):

```bash
export PATH="$HOME/.local/bin:$PATH"; export RADIX_API=https://nodes.sglang.io
radix assign verda-b300-fin-03-3       # free re-assign after every 4h lease lapse
ssh -i ~/.ssh/id_ed25519 -J ubuntu@95.133.252.66 bbuf@light-face-hides-fin-03-3
docker exec -it sglang_new bash        # CUDA_VISIBLE_DEVICES=7; flashinfer with sm_100f cubins preinstalled
```

Do not "upgrade" flashinfer casually in `sglang_new` — the
flashinfer/flashinfer-cubin pair must match, and TGV / cutlass_mla have no
sm_103 image. The k>7 unlock validation is a full 8-GPU server A/B
(see `docs/profile_evidence.md`).

Tier B. Follow `../../llm/docs/llm_kernel_optimization_rules.md` +
`../../llm/docs/llm_correctness_contract.md`.
