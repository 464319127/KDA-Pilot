# TileRT reference latencies — ≥3× ncu medians (B200)

Measured from the real `libtilert_dsv32.so` ops via **ncu** `gpu__time_duration.avg`
on an **idle B200** (GPU7), with `harness/sweep_ncu.py` (driver: `measure_ncu.py` →
`run_once.py`). Each number is the **median of 3 isolated runs**; `disp` = (max−min)/median.
Method: `benchmark_method.md`. Correctness of each op (golden vs real kernel) in
`../harness/ORACLE_RESULTS.md`.

> **Isolated vs in-graph.** These are *isolated single-launch* kernel times. TileRT runs
> the whole decode as one CUDA graph, so an op's *in-graph* per-call time (from the torch
> profiler, §16) can differ — e.g. PureMlaDsv32 is **35.2 µs in-graph** (52.8% of decode)
> but **~12 µs isolated** here (the isolated launch is just the attention compute over
> kv_len=2048; the in-graph cost folds in dependent waits/overlap).
>
> **Unified standard: kernels are compared ISOLATED single-launch, never in-graph.** The
> KDA target for a candidate kernel is the isolated median below, and a candidate is
> measured the same way. In-graph per-call numbers are engine-level reference only — an
> in-graph TileRT number is not comparable to a standalone candidate/open-source
> microbench (both pay the dependent-wait overhead in a real graph).

| op | TileRT kernel | seq | median µs | disp% | HBM% |
|---|---|---:|---:|---:|---:|
| rmsnorm | RMSNormExecutorImpl | 1 / 2 / 4 | 7.46 / 7.65 / 7.36 | ≤9.1 | ~0.1 |
| rmsnorm_quant | RMSNormQuantExecutorImpl | 1 / 2 / 4 | 8.74 / 8.96 / 9.02 | ≤3.9 | ~0.1 |
| head_proj | HeadProjExecutorImpl | 1 / 2 / 4 | **39.42** / 39.55 / 42.24 | ≤3.0 | **77.6 / 77.4 / 72.4** |
| rmsnorm_head_proj | RMSNormHeadProjExecutorImpl | 1 / 2 / 4 | 43.97 / 43.90 / 46.46 | ≤2.7 | 69.6 / 69.7 / 65.9 |
| rmsnorm_expert_proj | RMSNormExpertProjDsv32 | 1 / 2 / 4 | 8.22 / 9.02 / 10.62 | ≤8.2 | ~5 |
| projx_wis | ProjXWis | 1 / 2 / 4 | 6.11 / 5.98 / 6.59 | ≤7.9 | ~2 |
| projq_wqb | ProjQWkvbDevBHMMA | 1 / 2 / 4 | 5.92 / 6.21 / 7.84 | ≤9.7 | ~2.7 |
| projo_wkvb | ProjOWkvbDevBHMMA | 1 / 2 / 4 | 6.14 / 6.59 / 7.90 | ≤13.1 | ~2.6 |
| **flash_sparse_mla** (PureMlaDsv32 #1) | PureMlaDsv32 | 1 / 2 / 3 / 4 | **11.68 / 12.45 / 12.29 / 12.48** | ≤2.8 | ~2.6 |
| rotate | RotateExecutorImpl | 1 / 2 / 4 | 7.23 / 7.23 / 6.85 | ≤9.3 | ~0.1 |
| qkv_rope | QkvRopeExecutorImpl | 1 / 2 / 4 | 6.66 / 6.82 / 6.82 | ≤4.2 | ~0.03 |
| rmsnorm_kv | RmsnormKvExecutorImpl | 1 / 2 / 4 | 6.08 / 6.66 / 6.88 | ≤8.7 | ~0.03 |
| layernorm_rope_rotate (KV write) | LayernormRopeRotate | 1 / 2 / 4 | 6.88 / 7.04 / 7.17 | ≤6.4 | ~0.04 |
| rmsnorm_projq_wqb | RmsnormProjQWqbHMMA | 1 / 2 / 4 | 7.23 / 7.68 / 8.48 | ≤7.5 | ~10 |
| rmsnorm_projq_wqi | RmsnormProjQWqiHMMA | 1 / 2 / 4 | 8.42 / 8.86 / 9.44 | ≤8.0 | ~18 |
| rmsnorm_up_gate_silu | RMSNormUpGateSiLUDSv32 | 1 / 2 / 4 | 12.13 / 14.72 / 20.58 | ≤4.6 | 36 / 30 / 21 |

## Isolated standard — special cases
| op | isolated target | in-graph (ref only) | note |
|---|---|---|---|
| **SparseSelectMla** (`sparse_select_mla`) | **≈ 11.68 µs** | 35.4 µs | SAME flash_sparse_mla kernel as PureMla (GPU0 self-MLA, 16 heads vs worker's 20 → slightly less work); not separately swept — use PureMla's isolated median |
| FusedMoe (`fused_moe`) | **n/a** (floor = expert_proj 8.22 + up_gate_silu 12.13) | 22.4 µs (36.5% decode) | full 256-expert FP4 pack — not a standalone 1-call kernel; no isolated number until a dedicated FP4-pack harness exists |
| SparseIndex (`sparse_index`) | (isolatable via oracle) | 35.4 µs | DSA index scoring (GPU0, tcgen05) — distinct op from SparseSelectMla |
| comm (down/expert_down/unproj_o/eh_proj/padded allreduce, bcast/recv) | not isolatable | 8–11 µs | flag-based NVLink allreduce; needs 8-GPU peer setup |

## Validation
- `head_proj` s1 = 39.42 µs / 77.6% HBM here vs the prior single-run 39.2 µs / 78.4% —
  consistent, now with a 3-run median + 2.3% dispersion.
- All 48 isolated measurements have dispersion ≤13% (mostly ≤5%), i.e. stable.
- Raw per-run data: `harness/sweep_ncu.py` log; `harness/ORACLE_RESULTS.md` for correctness.
