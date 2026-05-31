# NCU + Roofline REPORT (final) — fused QK-Norm + RoPE on B200 (sm_100)

Authoritative round-0 roofline/diagnosis. Host `innomatrix-us-adc-smb200-0003`,
GPU 0 NVIDIA B200, verified idle. Extension built `-lineinfo`. Raw `.ncu-rep` in
REMOTE_KDA_DIR `profile/{cand_qwen4096_v1,_v2,_v3,base_qwen4096}/reports/`.

## Active bound: LATENCY-bound, not DRAM-bandwidth-bound

Cold-cache roofline (L2 flushed each launch; bytes = `4·N·H·D·2 + N·rope·4`):

| shape | impl | cold µs | BW TB/s | % of ~8 TB/s peak |
|---|---|---|---|---|
| qwen__4096 | baseline | 43.2 | 2.38 | 29.7 |
| qwen__4096 | **cand v3** | **35.7** | **2.88** | **36.0** |
| joyai__7904 | baseline | 90.4 | 2.91 | 36.4 |
| joyai__7904 | **cand v3** | **72.7** | **3.62** | **45.2** |
| qwen_edit__8424 | baseline | 75.0 | 2.82 | 35.2 |
| qwen_edit__8424 | **cand v3** | **60.5** | **3.49** | **43.7** |
| zimage__4128 | baseline | 50.5 | 2.55 | 31.9 |
| zimage__4128 | **cand v3** | **41.8** | **3.08** | **38.6** |

Both kernels reach only ~30–45% of HBM peak **even cold** → the limiter is
latency, not bandwidth: a small per-(token,head) in-place read-modify-write, a
warp/half-warp reduction dependency, and limited memory-level parallelism. The
SOL profile agrees (no single SOL saturated).

## NCU Speed-of-Light progression (qwen__4096, warm/L2-resident)

| metric | baseline | cand v1 | cand v2 | cand v3 |
|---|---|---|---|---|
| Duration (NCU, µs) | 59.0 | 62.6 | 60.1 | **50.1** |
| DRAM throughput % | 13.5 | 12.7 | 13.3 | **15.9** |
| Memory SOL % | 47.5 | 43.5 | 36.8 | 34.6 |
| Achieved occupancy % | 89.2 | 87.0 | 88.2 | 76.8 |
| Registers/thread | 32 | 28 | 28 | 32 |
| Uncoalesced flag | 28% (Est 23.8%) | 28% (Est 24.7%) | removed | removed |

## Diagnosis → design changes (KDA loop)

1. **v1→v2**: NCU flagged an uncoalesced cos/sin gather (strided scalar `__ldg`,
   28% excess sectors). Fixed with a coalesced `float2 __ldg` per lane. Large
   geomean 0.889×→0.926×. (The SGLang baseline still carries this pattern.)
2. **v2→v3** (the win): cold-cache roofline showed both kernels at only ~30% of
   peak → latency/low-MLP bound. Switched to **2-heads-per-warp + 128-bit
   (float4)** load/store (each lane owns 8 bf16, 16 lanes/head, half-warp RMS
   reduction, cos/sin float4 per lane). More bytes-in-flight per warp better
   hides the latency: cold BW ~30%→~40% of peak, NCU duration 60→50 µs, and the
   candidate now **beats the baseline on every shape**.

## Per-bucket bound & result

- **Large (4096–8424 tokens)**: latency-bound (cold ~36–45% of peak). v3 geomean
  **1.181×** over baseline (joyai 1.24×, qwen 1.16×, qwen_edit 1.18×, zimage 1.16×).
- **Tiny (19–195 tokens)**: launch/dispatch-bound — wall-clock latency is ~flat
  (~7.5–7.9 µs) regardless of token count (19 vs 195), the signature of
  launch-bound work; the kernel GPU time is a few µs. v3 geomean **1.079×** (the
  native path has lighter per-call dispatch than the tvm-ffi baseline). NCU of a
  tiny shape would only show under-occupancy and would not change the edit, so it
  is intentionally skipped per the profiling golden rule.

## Headroom / near-bound judgment

The candidate is ~40% of HBM peak (cold) and clearly ahead of the well-tuned
baseline. The remaining gap to peak is structural (in-place RMW + tiny per-token
work + reduction dependency); closing it further would need a different scheme
(multiple tokens/thread for ILP, persistent kernel) with diminishing returns and
added complexity/risk. Treated as near the attainable bound for this in-place
kernel structure, with a clear, evidence-backed win over baseline → promote.
