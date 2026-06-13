# Codex Goal Plan Generation Supplement

This file is a shared supplement for every `codex-goal-diffusion/*/prompt.md`.
It tightens the plan-generation contract while preserving the fair comparison:
Codex Goal plans may use the original task prompt and public diffusion rules,
but must not use prior task-local KDA optimization results.

## Fair Input Boundary

Allowed public inputs:

- The source task `prompt.md` named by the task wrapper.
- `codex-goal-diffusion/README.md` and launcher scripts.
- The public documents under
  `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/docs/`, especially:
  - `standalone_diffusion_benchmark.md`
  - `diffusion_kernel_rules.md`
  - `diffusion_correctness_contract.md`
  - `diffusion_benchmark_shape_coverage.md`
  - `standalone_diffusion_benchmark_template.py`
- Static interface/spec files explicitly needed to recover callable ABI or
  oracle semantics, provided they are not run records, benchmark outputs,
  candidate code, profiler logs, or optimization histories.

Forbidden task-local KDA result inputs remain forbidden:

- Do not read or use any existing task-local
  `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/kernels/<task>/baseline/`,
  `solution/`, `bench/`, `docs/`, `src/`, `tests/`, `profile/`, `ncu/`,
  `benchmark.csv`, `solutions.jsonl`, run logs, captured tensors, profiler
  traces, NCU reports, result summaries, dispatch summaries, candidate code,
  failure paths, or success paths.
- Do not copy KDA performance numbers, thresholds, dispatch tables, candidate
  designs, failed ideas, or final conclusions into the Codex Goal plan.

## Mandatory Public Docs Extraction

The plan-generation turn must read the public diffusion docs above directly,
not only the source task prompt's short task card. The resulting `plan.md` must
show how those docs constrain the future Goal.

At minimum, extract:

- From `diffusion_benchmark_shape_coverage.md`: the relevant task-family
  production shapes, layout classes, dtypes, entry points, scalar parameters,
  current-preset audit status, and any blocked/no-call preset requirements.
- From `diffusion_kernel_rules.md`: the explicit permission to use
  shape-specialized kernels, template variants, autotune tables, and
  dispatchers when evidence shows different workload buckets need different
  tradeoffs.
- From `standalone_diffusion_benchmark.md` and the benchmark template: the
  fixed timing policy, workload-freeze rule, symmetric baseline/candidate ABI,
  output-preallocation rule, interleaved CUDA-event measurement, provenance,
  and remeasure-on-method-change requirements.
- From `diffusion_correctness_contract.md`: canonical regression rows, dtype
  coverage, tolerances, NaN/Inf checks, poisoned-output checks, and negative
  validation cases relevant to the task family.

If a public doc does not cover the task family, the plan must say so explicitly
and fall back to the source prompt and static ABI/spec files without inventing
diffusion-specific semantics.

## Required Shape-Dispatch Planning

Every generated `plan.md` for a multi-shape task must include a workload-driven
dispatch/search section. This is a planning requirement, not prior KDA result
reuse.

The plan must require the future Goal to:

- Derive initial shape/layout/dtype/entry-point bucket hypotheses from the
  public shape coverage before choosing a nontrivial candidate strategy.
- Treat a single universal kernel as a hypothesis to prove, not as the default
  assumption.
- Benchmark and profile representative buckets before promoting or rejecting a
  universal candidate.
- Consider shape-specialized kernels, template variants, autotune tables, or
  dispatchers when public workload coverage shows materially different shapes,
  layouts, dtypes, ranks, affine/gate modes, or output counts.
- Record each bucket hypothesis, candidate attempt, correctness result,
  benchmark result, profile/roofline evidence, and promote/reject decision in
  a fresh task-local search DAG or solutions ledger.
- Write `docs/dispatch.md` when dispatch or specialization is used, including
  bucket conditions, selected implementation, per-bucket evidence, and why that
  bucket does not use another implementation.
- If the final candidate is universal, document per-bucket evidence showing why
  specialization was not needed.

The plan's Acceptance Criteria must make this enforceable. A final Goal result
should not be accepted merely because one candidate improves the overall
geomean while hiding large unexplained bucket regressions or while skipping
the dispatch analysis required by the public rules.

## Success Standard

Keep the greenfield success standard:

- correctness over all required production and regression rows;
- measured improvement over the same-environment recovered baseline, unless an
  evidence-backed no-go is reached;
- complete benchmark/profiling/roofline evidence;
- no use of prior task-local KDA optimization artifacts.

Do not add KDA promoted speedups as thresholds. The point of this supplement is
to preserve public prompt knowledge, especially shape coverage and dispatch
permission, while keeping the optimization search independent.

## Search Rigor and Stopping Discipline

(This section OVERRIDES any weaker stopping or acceptance wording in this file
or in any task prompt. It constrains HOW rigorously the future Goal must search
and WHEN it is allowed to stop. It must not be read as prescribing any
particular implementation language, kernel technique, memory layout, or
optimization lever — those remain for the Goal to discover from first
principles, profiling, and public prior art.)

### Profiling Gate

- As soon as a correct candidate exists, the Goal MUST collect profiler and/or
  Nsight Compute evidence on representative production buckets BEFORE any
  promote, reject, or stop decision.
- Each profiling finding must be bound to a concrete next action: a further edit,
  a rejection with a stated reason, or a hardware-bound proof for that bucket.
- "The candidate passed, so no profiling was taken" is an explicit failure of
  this gate. Profiling collected only as decoration, with no decision tied to it,
  also fails.

### No Bucket Abandonment

- The Goal may NOT leave a required production bucket at baseline-equivalent
  performance by labeling it "parity within noise", "out of scope", or "not the
  bottleneck of my hypothesis".
- For every required production bucket, the final result must show EITHER (a) a
  measured improvement over the recovered baseline, OR (b) per-bucket roofline
  evidence — bytes moved, useful work, achieved vs peak bandwidth and/or FLOP/s,
  and the named active bound — proving that bucket already sits at its hardware
  limit. Only a bucket proven at its bound may be routed to a fallback.
- A headline geomean that hides unexplained per-bucket regressions or unprofiled
  buckets is not an acceptable result.

### Iteration Depth

- The bounded optimization loop must run multiple distinct candidate iterations,
  not a single attempt promoted on first correctness. Each iteration is logged in
  the task-local search DAG / solutions ledger with: hypothesis, the change made,
  correctness result, benchmark result, profiling evidence, and the
  promote/reject reason.
- The loop continues while any profiled production bucket still shows headroom
  against its roofline bound and a defensible next hypothesis remains.

### Stopping Conditions (replaces softer completion standards)

- Stop with SUCCESS only when: correctness passes on all required production and
  regression rows; the production geomean improves over the recovered
  same-environment baseline; and every required production bucket satisfies the
  "No Bucket Abandonment" rule above (improved, or proven at its hardware bound).
- Stop with NO-GO only after ALL of: the baseline is recovered; at least one
  correct candidate has been built; profiling/NCU evidence has been collected on
  EVERY required production bucket; and a named hardware bound shows no remaining
  headroom on every bucket. A NO-GO is NEVER permitted while any required
  production bucket is still unprofiled, or while no real optimization iteration
  beyond baseline recovery has been attempted.

### Neutral Timing Policy

- The timed region must be defined once and applied IDENTICALLY to baseline and
  candidate: the same inputs, the same output-passing policy, the same CUDA
  stream, and the same set of operations included in / excluded from timing.
- The plan must state explicitly and symmetrically whether the production host
  call path (any Python / framework / dispatch wrapper of the named entry point)
  is inside or outside the timed region, and apply that one choice to both sides.
  The plan must NOT recommend, hint at, or pre-judge any specific way of changing
  that cost — it only fixes the measurement so the comparison is fair.
