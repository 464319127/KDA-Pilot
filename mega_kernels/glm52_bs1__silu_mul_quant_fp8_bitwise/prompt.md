# KDA Prompt: glm52_bs1__silu_mul_quant_fp8_bitwise

Target GPU: NVIDIA B300 (sm_103). Fuse `silu_and_mul` + per-token-group(128)
fp8 quant between the two triton MoE GEMMs — small (~2% of round, ~0.25 ms)
but it is the last **tier A (bitwise)** win available on the main-model MoE
path, alongside the already-landed `tiny_align` (289.9→292.6 tok/s with the
same contract).

## Problem (FIXED)

Between gemm1 and gemm2 of sglang's `fused_experts_impl` on GLM-5.2 decode:

```
in : cache1 [M*8, 512] bf16   (gemm1 out: [gate | up] halves, M ∈ 1..8)
out: cache2_q [M*8, 256] fp8e4m3 + scale [M*8, 2] fp32
math: y = silu(cache1[:, :256]) * cache1[:, 256:]
      per 128-group: s = max|y_group| / 448; q = round(y_group / s)
```

Baseline: sgl `silu_and_mul` kernel (~3.1 µs) + sgl
`per_token_group_quant_fp8` (~1.7 µs) at M=8, two launches (plus their gap
inside the decode graph). **Success: one launch, ≤ 2.0 µs at M=8, and
BITWISE-identical output (q AND s) to the baseline pair** — elementwise math
with per-group reductions has a canonical order; replicate the baseline's
exact rounding/scale formula (copy it from the sgl kernel source, do not
re-derive: `448.0` clamp, `1e-10` guards, rounding mode all matter).

## Why bitwise matters here

Measured on this deployment: the deep MTP chain's full-accept mode (42% of
rounds, carries the throughput) dies under ANY main-model numerics shift
≥ ~1e-3 elementwise, while GSM8K stays flat — even relocating a single
bf16 round-trip collapsed accept 5.16→3.3. Only bitwise-identical rewrites
are deployable without an accept A/B. That makes this a *verification*
exercise as much as a perf one: the correctness harness must assert
bit-equality (`torch.equal` on the uint8 view and the fp32 scales) across
the full shape grid and random seeds, poison-buffer included.

## Approach notes

- [M*8, 512] bf16 = 64 KB at M=8: one block per 4-8 rows, vectorized 16B
  loads, warp-reduce the two 128-group absmax per row-half, fp32 silu path
  copied verbatim from the baseline source (order of operations included).
- The win is mostly the removed launch+gap (~2.8 µs of the ~4.8 µs pair);
  don't over-optimize the arithmetic at the cost of bit-parity.
- Integration point: mini-sglang calls `fused_experts_impl`; the fused op
  replaces the act+quant pair inside a copied `_fused_moe_kernel_sequence`
  (pattern proven by `tiny_align`, see mini-sglang
  `python/minisgl/kernel/moe_glue.py`).

## Hardware access (B300)

Full runbook: `../docs/b300_access.md`. Single idle GPU suffices; the bitwise
check runs anywhere with sgl_kernel, the perf target is B300-specific:

```bash
export PATH="$HOME/.local/bin:$PATH"; export RADIX_API=https://nodes.sglang.io
radix assign verda-b300-fin-03-3       # free re-assign whenever the 4h lease lapses
ssh -i ~/.ssh/id_ed25519 -J ubuntu@95.133.252.66 bbuf@light-face-hides-fin-03-3
docker exec -it sglang_new bash        # CUDA_VISIBLE_DEVICES=7; sgl_kernel baselines preinstalled
```

Sync via base64-over-ssh; check `nvidia-smi` for the glm_pd tenant first.

Follow `../../llm/docs/llm_kernel_optimization_rules.md` +
`../../llm/docs/llm_correctness_contract.md`. Baseline = sglang's
`silu_and_mul` + `sglang_per_token_group_quant_fp8` sources (copy, record
SHAs).
