# PR evidence — GLM-5.2 `grouped_topk` warp-per-token kernel (B200)

Evidence package for upstreaming the warp-per-token `grouped_topk` candidate.
**This is a kernel-level optimization**; it is justified by the isolated kernel
benchmark below (the standard basis for an SGLang kernel PR), with an honest
end-to-end scope statement.

## 1. Large-prefill kernel speedup (the headline result)

Isolated baseline-vs-candidate microbenchmark on **NVIDIA B200**, the
project-standard harness (`bench/benchmark.py`, CUDA-events, inner-loop
amplification, interleaved A/B, 21 trials, `--no-isolated`,
`--target-sample-us 5000`). Shapes are the GLM-5.2 production routing config
(`E=256, topk=8, num_expert_group=1, topk_group=1, renormalize, scaling=1.0`)
swept across the **batched-prefill token range GLM-5.2 actually produces** (up to
`chunked_prefill_size=16384`). Workloads: `bench/workloads_largeN.json`; raw:
`bench/results_largeN.jsonl`.

| N (tokens) | baseline µs | candidate µs | speedup |
|---:|---:|---:|---:|
| 1536  | 12.30 |  8.20 | 1.50× |
| 2048  | 12.30 | 10.25 | 1.20× |
| 2560  | 16.40 | 10.26 | 1.60× |
| 3072  | 16.40 | 10.26 | 1.60× |
| 4096  | 20.52 | 12.30 | 1.67× |
| 6144  | 28.59 | 14.35 | 1.99× |
| 8192  | 34.83 | 18.44 | 1.89× |
| 12288 | 44.95 | 22.57 | 1.99× |
| 16384 | 54.31 | 28.70 | 1.89× |

**Equal-weight geomean = 1.68× (min 1.20×, max 1.99×).** The speedup **grows with
N** (≈2× at N≥6144): the larger the prefill batch, the more the candidate's
warp-per-token packing wins over the baseline's one-block-per-token layout that
leaves 7/8 warps idle during top-k selection.

## 2. Full captured production set + independent reproduction

On the full captured GLM-5.2 production shape set (`bench/workloads.json`, 43
production rows, N=2..3769): **equal-weight geomean 1.22×, prefill-regime 1.30×,
decode parity 1.000×, no regression (min 0.999×)** — see `docs/results.md`.
Independently reproduced on a second B200 host (`ion-b200`) to 4 decimals
(1.2203× equal-weight, 1.674× max).

## 3. Correctness (preserved, exact)

`bench/correctness.py`: **1695 checks, 0 failures** — exact ordered
`topk_indices` + `topk_values` (fp32 `atol=rtol=1e-5`) vs the recovered baseline
and an independent oracle, across the production shapes (×2 seeds) and an edge
grid (ties→smaller index, negative/saturating bias, N=0, max N, renormalize off,
off-domain fallback, non-contiguous/non-fp32 rejection). Decode/small-N and all
off-domain inputs **dispatch to the bit-identical recovered baseline kernel**, so
they are correct by construction with no regression. See `docs/dispatch.md`.

## 4. Why it wins (NCU roofline)

Neither regime is compute- or memory-bound; the kernel is occupancy/latency
bound (per-token `__expf` SFU + serial top-k). NCU (`--set basic`, B200) on the
representative large-N kernel: achieved occupancy **27%→33%**, active warps
**17→21**, block count **4× lower** (3769→943 blocks; 3.18→0.53 waves), register
pressure 40/thread with **no spill**. The candidate makes every warp productive
(one warp per token, register-resident, no shared memory) and packs several
tokens per CTA. Full analysis in `docs/results.md` §Evidence-backed analysis.

## 5. Honest end-to-end scope

This kernel is a **microsecond-scale MoE router**; GLM-5.2 prefill is dominated
by attention (DSA) + 256-expert GEMMs (2–3 orders of magnitude larger). A direct
end-to-end serving A/B (GLM-5.2-FP8, TP=8, B200, vanilla; summary/chat ×
low/mid/high) measured:

- **conc=1 (clean signal): TTFT parity within ±2%** (summary 53.31→53.42 ms,
  chat 46.35→45.43 ms) — single-request prefill N≈1000 is the candidate's parity
  zone; the win starts at N≥1464.
- **mid/high concurrency: dominated by RadixCache + scheduling variance**
  (2–25× run-to-run swings on the *same* kernel) — far larger than any router
  effect, so no reliable e2e signal at sub-1%.

**Estimated e2e benefit ≈0.05% (≪1%)** — the router is <1% of prefill. So this
PR is justified on **kernel-benchmark grounds** (a correctness-preserving 1.2–2×
on large-prefill MoE routing, decode parity, no regression), **not** on an e2e
TTFT claim. The win is real and grows with batched-prefill size; large-batch /
high-token-throughput prefill workloads benefit most, online single-stream
decode is unaffected (parity by construction).

## Reproduce

```bash
# isolated kernel benchmark (no model/server needed), on an idle B200:
cd <task dir>
CUDA_VISIBLE_DEVICES=<idle> TVM_FFI_CUDA_ARCH_LIST=10.0 \
  python bench/benchmark.py --workloads bench/workloads_largeN.json \
  --device cuda:0 --target-sample-us 5000 --num-trials 21 --no-isolated \
  --out bench/results_largeN.jsonl
python bench/correctness.py   # 1695 checks
```
