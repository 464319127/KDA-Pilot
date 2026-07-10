# Dispatch — mega_kernel task 01 (mnnvl_ar_fused)

Folder-contract artifact: the shape-specialization dispatch table, fallback
policy, and serving gate behavior for the promoted deliverable. Evidence
links at the bottom.

## Dispatch table (standalone entry `mnnvl_ar_fused_opt`)

Host-side scalar compares only; no hot-path syncs; selection happens per
call inside `oneshotArFusedConstDispatch`
(`solution/mnnvl_ar_fused/csrc/mnnvl_ar_fused_opt.cuh`).

| Regime | Kernel | Per-regime result (jsonl-backed, 25 trials, same-session pairing) |
|---|---|---|
| T=6, H=6144, world=8, bf16, oneshot, rmsnorm, exact-fit geometry | `oneshotArFusedNormConstKernelBA<8,bf16,6,6144>` (block-arrival, ADOPTED round 1) | **1.0563** vs ported baseline (noise 0.36%); provenance-complete rerun **1.0546** (noise 1.60%) |
| T=1, H=6144, world=8, bf16, oneshot, rmsnorm, exact-fit geometry | `oneshotArFusedNormConstKernelBA<8,bf16,1,6144>` | **1.0884** vs ported baseline (noise 0.46%); provenance-complete rerun **1.0888** (noise 0.33%) |
| frozen shapes but non-exact-fit geometry (defensive; cannot occur for H=6144 on this arch) | `oneshotArFusedNormConstKernel` (cluster-arrival) | 1.0126 / 1.0230 vs ported baseline (round-1 records; superseded candidate kept only as the guarded fallback) |
| any other (shape, dtype, world, pattern) | generic verbatim `oneshotAllreduceFusionKernel` dispatch | parity within +-3% of the flashinfer original (P0 gate) |

Launch geometry for the specialized rows comes from the SAME
`adjustGridConfig` as the generic path (grid 24 x block 192 at T=6) — pinned
so the norm reduction tree stays byte-identical.

## Serving-side routing (env-gated, layered fallbacks)

Route lives in `flashinfer_comm_fusion.py` (applied by
`solution/serving/sglang_patcher.py`); module in
`sglang/jit_kernel/mnnvl_ar_fused.py`.

| Condition | Path |
|---|---|
| `SGLANG_JIT_MNNVL_AR` unset/0 (DEFAULT) | stock flashinfer route — byte-inert, measured twice (40/40 records identical) |
| flag on, backend != mnnvl, or fp32_acc, or `use_oneshot is False` | stock flashinfer route |
| flag on, `input_tensor.dtype != bf16` (jit module is bf16-only; stock handles fp16/fp32) | stock flashinfer route (guard added post-review, patch v2-dtype-guard) |
| flag on, `trigger_completion_at_end=True` (jit wrapper has no end-of-kernel completion; stock forwards the flag) | stock flashinfer route (guard added post-review, patch v3-trigger-guard) |
| flag on, payload > oneshot threshold (1 MiB; e.g. prefill) | stock flashinfer route (twoshot) |
| flag on, workspace missing any required attribute | stock flashinfer route (defensive) |
| flag on, oneshot decode regime | jit module, generic port symbol |
| + `SGLANG_JIT_MNNVL_AR_OPT=1` | jit module, `mnnvl_ar_fused_opt` (table above; uncovered shapes fall back to the generic port INSIDE the same call) |

Correctness is never lost: every uncovered combination reaches the verbatim
port (bit-exact vs the deployed binary incl. the adversarial value zoo) or
the stock flashinfer path.

## Evidence links

- Specialization A/B + noise + stability: `docs/results.md` ("P1
  specialization", "25-trial confirmation"); raw records
  `bench/results.jsonl`.
- Bit-exactness incl. value zoo: `docs/results.md` ("P0 Port —
  bit-exactness"); `bench/value_zoo_ab.py`.
- Serving gate + promote (sanity/official/per-call, all 40/40
  byte-identical): `docs/results.md` serving tables; raw summaries/records
  `docs/serving_runs/` (adopted-kernel runs: `results_task01_ba_*`);
  per-call profiles: `percall_profile_task01_ba_tp0.txt` (adopted,
  7.6 us/call) and `percall_profile_task01_opt_tp0.txt` (superseded
  cluster-arrival candidate); pre-task baseline official (376.06):
  `docs/serving_runs/baseline_official_pretask/`.
- Value-zoo bit-exactness of the adopted entry:
  `docs/profiler_evidence/zoo_fi_vs_opt_adopted.txt`.
- Epilogue-fold trace adjacency:
  `docs/profiler_evidence/trace_adjacency_task01_ba.txt`.
- NCU breakdown: `docs/profiler_evidence/ncu_{jit,opt}_t6_details.txt`.
- Route-closure evaluations (PDL-with-predecessor, fence/flag granularity,
  epilogue-fold feasibility): `docs/results.md` ("P1 route evaluation
  closure") and `RESULTS_SM103.md`.
