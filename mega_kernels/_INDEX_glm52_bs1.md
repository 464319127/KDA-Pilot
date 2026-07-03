# glm52_bs1 — fixed-shape kernel task selection (B300, TP=8, k=7 MTP)

- Baseline e2e: **292.6 tok/s** (mini-sglang `glm5.2-support` @ `a26fd6f`,
  accept 5.16, round ≈ 17.6 ms). Every 100 µs cut from the round ≈ **+1.7 tok/s**.
- Selection: measured round anatomy (see `README.md`), single-GPU or NVLink
  kernels only, fixed decode shapes.
- Tier = accept-preservation tier (A bitwise / B accept-gated), see README rule 3.

| pri | task | family | round share | tier | headroom (est.) |
|---|---|---|---:|---|---|
| P0 | `glm52_bs1__fused_moe_decode_fp8` | moe | ~23% | B | ~1.3 ms/round → +20 tok/s |
| P0 | `glm52_bs1__oneshot_allreduce_bf16` | comm | ~11% | B (comm reorder ≈ NVLS band, validate) | ~0.9 ms → +15 tok/s |
| P1 | `glm52_bs1__skinny_gemm_bf16_tc` | quant_gemm | ~14% | B | ~0.6 ms → +10 tok/s |
| P1 | `glm52_bs1__mtp_chain_megakernel` | spec_decode | ~11% | B (MTP side: safe) | ~0.8 ms → +13 tok/s |
| P2 | `glm52_bs1__mla_decode_bs_consistent` | attention | ~8% | B + consistency unlock | ~0.4 ms → +7; bs-consistency could unlock k>7 (+8-10%) |
| P2 | `glm52_bs1__silu_mul_quant_fp8_bitwise` | moe glue | ~2% | A (bitwise required) | ~0.25 ms → +4 tok/s |

## Known results to not repeat (all measured on this exact setup)

- CUDA-core skinny GEMV loses to cuBLAS/nvjet at every decode dense shape
  (`python/minisgl/kernel/csrc/jit/skinny_gemv.cu` in mini-sglang tree,
  unwired). Tensor cores are mandatory for `skinny_gemm_bf16_tc`.
- vLLM-style one-shot allreduce (LD/ST over IPC buffers, no multimem):
  20.4-32 µs vs NCCL NVLS 12.1 µs — must use NVLS multimem.{ld_reduce,st}.
- Custom fused MoE (`moe_decode.cu`, bf16-act x fp8-dequant, cp.async
  double-buffered): M=1 19.3 µs beats triton 41.7; M=8 66 µs still above
  tuned triton+glue ~55 µs and the 37 µs roofline. g1 (gate_up) is the gap:
  4.7 TB/s achieved vs ~8 TB/s needed. Deployed to the MTP layer only
  (raised accept 5.0→5.2); main-model deploy collapsed accept (tier B
  lesson).
- trtllm fp8 block-scale MoE (flashinfer): 3x noisier intermediates, accept
  drop, net loss. DeepGEMM masked grouped: 80.7 µs @M=8, loses.
- Triton fused_moe config sweep: 16x64x128 w4s4 already optimal in-server;
  standalone default config (64x128x128) resolves for all M<=16 — config is
  NOT the k>=8 accept-collapse cause.
- PDL sprinkled on existing mini kernels: -7% e2e (serialization stalls);
  PDL only pays inside purpose-built chains.
- k>=8 (verify batch >=9) collapses accept 5.16→3.2-3.4 with GSM8K unharmed:
  replaying verify/extend on the bs-9 graph at unchanged k=7 semantics
  reproduces it — per-batch-size kernel numerics (nvjet M-dispatch,
  trtllm-gen bs scheduling) shift the model's numerics style. This is why
  `mla_decode_bs_consistent` lists the consistency unlock.

## Shared benchmark seeds

Copy from mini-sglang tree / this repo when building `bench/`:

- CUDA-graph microbench harness pattern: `moe_decode_test.py` /
  `moe_prof.py` style (graph-of-20, 50 replays, per-kernel split via
  torch.profiler) — reproduced in each task's `bench/README`.
- In-server A/B for tier-B deploys: `bench_bs1_30000.py` (3x256-token greedy
  essay prompt) + GSM8K harness `/data/bbuf/glm_gsm8k_fixed.py`
  (`--parallel 1`, spec only engages at bs=1) + `MINISGL_SPEC_LOG=1` accept
  histogram (bimodal full-accept bin must survive).
