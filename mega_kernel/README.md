# mega_kernel — GLM-5.2 bs=1 low-latency kernel campaign (8×B300 / sm_103)

Isolated kernel-optimization tasks for the GLM-5.2 FP8 bs=1 MTP(5-1-6) decode
path. Frozen production shapes come from the live serving profile on
`light-face-hides-fin-03-1` (rx devbox `glm52-bs1-opt`); each task follows the
KDA-Pilot rules: copied upstream baseline, symmetric benchmark, correctness
gate, NCU evidence, promotion only on `geomean > 1.0` with no production row
`< 0.97`. Promoted kernels flow back into serving through env-gated dispatch
hooks (see `integration/`).

## Serving context the tasks must respect

- One MTP iteration (accept ≈ 3.95) = draft graph (4×M=1 fwd) + target-verify
  graph (M=6 through 78 layers) + draft-extend graph (M=6, 1 layer).
- Current official e2e: **354.1 tok/s** (11.15 ms/iter = GPU ~9.3 ms + ~1.8 ms
  CPU exposure). Baseline chain: 307.17 → 316.05 (phase-1 sync fixes) → 354.10
  (bf16-dense + FP8 deferred finalize).
- Everything on the hot path replays inside CUDA graphs → **benchmark in-graph
  and with cold L2** (rotate ≥48 weight copies; replay-same-weight measures L2,
  not DRAM — this trap is documented in `glm52_blog_bench/REPORT_OPT.md`).
- GPU budget per iteration (kernel-duration deflated): MoE GEMMs 3.2 ms
  (~82% of weight-BW bound), dense GEMMs 2.8 ms (bf16 cuBLAS today), fused AR
  1.4 ms (mnnvl oneshot 8.7 µs × 160, payload 73 KB), attention 1.1 ms
  (fmhaSm100f 9.1 µs × 83), MoE aux 0.65 ms, norms/misc ~1.1 ms.
  In-graph inter-kernel gaps (<10 µs class) total only ~0.4-0.5 ms/iter — PDL
  is a garnish, not the main course (task 05 quantifies per-pair anyway).

## Task index (value-ordered)

| Task | Target | Baseline | Budget → floor | Expected e2e |
|---|---|---|---|---|
| tasks/01 (dense_fp8_gemm_bs1) | M∈{1,6} w8a8 block-FP8 dense GEMM | DeepGEMM fp8+quant AND cuBLAS bf16 (both) | 2.8 → ~1.3 ms | −1.4 ms/iter |
| tasks/02 (ar_norm_fused_bs1) | 8-rank oneshot AR(+add+rmsnorm), 73 KB | flashinfer mnnvl `oneshotAllreduceFusionKernel` 8.7 µs | 1.4 → ~0.5 ms | −0.8 ms/iter |

Both promoted: GPU 9.3 → ~7.1 ms → iteration ~8.9 ms ≈ **~445 tok/s** (accept 3.95).

Future (deliberately out of scope for now): MoE aux fusion (−0.5), DSA sparse
decode q≤6 port (−0.35), PDL audit (−0.2-0.4).

## Ground rules

1. **Shapes are frozen** (see each task's `SHAPES.md`); per-TP-rank, TP=8.
2. **Two benchmark modes, both mandatory**: (a) cold-L2 in-graph replay
   (production-faithful), (b) eager per-call (debug only, never the headline).
3. **Correctness**: fp32 oracle, rel err < 2e-2 for GEMM-class, bitwise for
   memory-movement class; adversarial cases (ragged tails, scale ramps,
   negative dispatch) required before promotion.
4. **NCU evidence** for every claimed win (`ncu-report-skill` conventions).
5. **Integration is env-gated**: each task lists its serving hook; e2e
   validation = 40-task × 1-run sanity on the devbox (greedy determinism +
   accept-len watch) before any 3-run official number.

## Prior art to reuse (from `对比.md` / earlier KDA-Pilot campaigns, B200)

- `deep_gemm_fp8_fp8_bf16_nt` M=1 decode GEMV: kernel-honest **1.356×**
  (peak 1.93×) vs DeepGEMM — port to sm_103 for the draft path (task 01).
- M>1 CUDA-core attempts are a confirmed dead end (two independent runs:
  0.011-0.44×); M=6 requires CUTLASS SM100 blockwise tensor-core work.
- Unified attention native split-KV flash-decode, B=1: **1.87×** geomean vs
  fmhaSm100f — port + extend q 1→6 (task 04).
- `per_token_group_quant` promoted 1.115× — low relevance now (bf16-dense
  removed most quant calls; only MoE input quant remains, 80/iter).
- fp8_bmm eager wins were host-dispatch-bound — NOT applicable in-graph.
