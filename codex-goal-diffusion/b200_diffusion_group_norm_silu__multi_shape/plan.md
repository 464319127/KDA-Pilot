# b200_diffusion_group_norm_silu__multi_shape Codex Goal Plan

## Goal Description

Produce a greenfield, evidence-backed B200 optimization result for
`b200_diffusion_group_norm_silu__multi_shape` inside this Codex Goal task
directory. The later Goal is launched from this directory with:

```text
/goal follow the instruction in plan.md
```

The Goal is complete only when the task has recovered a fresh local baseline,
built a matching local candidate ABI, generated a complete workload and
correctness harness, measured candidate versus baseline on B200 under the
standard standalone benchmark policy, and written an evidence-backed final
report. If no correctness-preserving speedup is defensible, the Goal may stop
only with a documented no-go that includes baseline evidence, attempted
candidate evidence, profiling or roofline evidence, and the blocker that would
unlock further progress.

This is a new greenfield Goal. The source KDA prompt and shared docs define
the task and constraints; prior KDA optimization processes, candidate code,
benchmark numbers, profiler traces, logs, and conclusions are not input facts.

## Source Task Summary

- Task slug: `b200_diffusion_group_norm_silu__multi_shape`.
- Target GPU: NVIDIA B200.
- Source task prompt:
  `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/kernels/b200_diffusion_group_norm_silu__multi_shape/prompt.md`.
- Later Goal cwd:
  `/Users/bbuf/工作目录/Common/KDA-Pilot/codex-goal-diffusion/b200_diffusion_group_norm_silu__multi_shape`.
- Required local artifact layout:
  `baseline/`, `solution/`, `bench/`, and `docs/` under the later Goal cwd.
- Target SGLang diffusion entry points to copy into `baseline/`:
  `sglang.jit_kernel.diffusion.triton.group_norm_silu:triton_group_norm_silu`
  and
  `sglang.jit_kernel.diffusion.group_norm_silu:apply_group_norm_silu`.
- Operation semantics:
  fused GroupNorm followed by SiLU, equivalent to
  `silu(group_norm(x, num_groups, weight, bias, eps))`.
- Production workload family:
  HunyuanVideo VAE GroupNorm+SiLU shapes on B200.
- Production dtype and parameters from the shared shape coverage contract:
  fp16, `num_groups=32`, and `eps=1e-6` for the Triton entry point.
- Production shape coverage must include both target entry points and both
  contiguous and non-contiguous cases when observed for the target architecture.
  The retained union includes channels `512`, `256`, and `128`; temporal depths
  `2`, `3`, `5`, `9`, and `17`; and spatial pairs including `12x10`,
  `12x32`, `24x20`, `24x64`, `32x10`, `32x32`, `48x40`, `48x128`,
  `64x20`, `64x64`, `96x80`, `96x256`, `128x40`, `128x128`, `256x80`,
  and `256x256`.
- Baseline type:
  copied upstream SGLang source from the resolved latest upstream `main`
  commit, exposed through a local task-owned ABI.
- Candidate boundary:
  task-local optimized implementation in `solution/` with the exact same ABI,
  argument ordering, output allocation policy, stream behavior, and comparable
  wrapper overhead as the copied baseline. Candidate implementation may use
  CUDA/C++/Triton/Python only if the baseline and candidate comparison remains
  fair and local.

## Final Goal State

When the later Goal succeeds, the directory contains these files and evidence:

- `baseline/` contains copied upstream SGLang source files for the target entry
  points plus local baseline binding/build code.
- `solution/` contains the final candidate implementation plus local
  binding/build code.
- `bench/workloads.json` contains the frozen production workloads and
  regression rows, with shapes, dtypes, strides/layouts, scalar parameters,
  tolerances, seeds, function selectors, and headline-score flags.
- `bench/benchmark.py` starts from
  `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/docs/standalone_diffusion_benchmark_template.py`.
- `bench/adapter.py` implements `make_case`, `call_baseline`, and
  `call_candidate` without timed-path output allocation.
- `bench/correctness.py` implements the oracle, positive tests, negative
  validator tests, NaN/Inf checks, stale-output checks, and regression grid.
- `bench/results.jsonl` or timestamped equivalent records the final benchmark
  with full provenance.
- `docs/baseline_source.md` records upstream repository URL, branch, resolved
  commit, resolution time, copied files, relevant versions, local ABI, and
  compile flags.
- `docs/benchmark_method.md` records workload freeze point, benchmark command,
  timing settings, tolerance policy, ABI fairness notes, and any method changes
  that forced remeasurement.
- `docs/run_log.md` records host, GPU id, GPU model, before/after GPU state,
  commands, failures, fixes, and major decisions.
- `docs/benchmark_preset_audit.md` exists if any current preset is missing
  from the retained rows and must be explained by fresh capture or live
  no-call proof.
- `docs/research.md` or `docs/draft.md` records KernelWiki/prior-art ideas,
  adopted/rejected decisions, and the fresh search DAG or solution-attempt
  ledger for this Goal.
- `docs/results.md` contains the final per-workload baseline-versus-candidate
  table, equal-weight production geomean, arithmetic mean, command lines,
  provenance, profiling or roofline explanation, conclusion, residual risks,
  and either success or no-go status.
- `docs/dispatch.md` exists if shape specialization, autotune buckets, or a
  dispatcher is used.

## Acceptance Criteria

- AC-1: Baseline recovery is fresh, local, and provenance-complete.
  - Positive Tests (expected to PASS):
    - `docs/baseline_source.md` names the upstream SGLang repository URL,
      branch `main`, resolved commit SHA, resolution timestamp, copied files,
      local ABI, dependency versions, and compile flags.
    - `baseline/` contains only task-local copied baseline source and local
      binding/build files needed for the two target entry points.
    - The baseline can be called from the task-local harness without importing,
      patching, monkey-patching, or installing into SGLang at correctness or
      benchmark runtime.
  - Negative Tests (expected to FAIL):
    - A baseline copied from a previous KDA task, stale unknown checkout, or
      undocumented commit is rejected.
    - Any runtime path that imports live SGLang modules for correctness or
      benchmark timing is rejected.

- AC-2: Baseline and candidate expose the same local ABI and fair wrapper cost.
  - Positive Tests (expected to PASS):
    - `bench/adapter.py` calls baseline and candidate through matching local
      entry points with identical argument ordering, output tensors passed in,
      stream behavior, and output allocation policy.
    - If either side uses CUDA/C++, CUDA launches use PyTorch current stream,
      for example `at::cuda::getCurrentCUDAStream()`.
    - Compile and build flags that can affect numerics or code generation are
      symmetric and documented.
  - Negative Tests (expected to FAIL):
    - Timing a heavy Python/Triton wrapper for baseline against a lean direct
      CUDA candidate without equivalent wrapper overhead is rejected.
    - `--use_fast_math` on only one side, or without matching upstream baseline
      usage, is rejected.

- AC-3: Workload coverage is complete and frozen before tuning.
  - Positive Tests (expected to PASS):
    - `bench/workloads.json` includes all retained live production rows for
      `diffusion_group_norm_silu__multi_shape` and B200, covering both target
      entry points, required HunyuanVideo VAE fp16 rows, observed layouts, and
      headline-score metadata.
    - Because this family has many rows, workloads are generated from a fresh
      live HunyuanVideo capture, or from an explicitly permitted retained raw
      shape-capture source that is not a task-local KDA optimization result,
      with provenance recorded.
    - Every current benchmark preset missing from the retained rows has either
      a fresh capture row or a live no-call/runtime-blocked note in
      `docs/benchmark_preset_audit.md`.
  - Negative Tests (expected to FAIL):
    - A hand-written reduced production list is rejected.
    - Silently skipping a production workload because baseline, candidate,
      compilation, correctness, or runtime fails invalidates the benchmark.
    - Changing workloads, tolerances, scoring, or timing policy after tuning
      starts without deleting stale results and remeasuring both sides is
      rejected.

- AC-4: Correctness uses an independent oracle and the canonical regression
  grid.
  - Positive Tests (expected to PASS):
    - `bench/correctness.py` checks the oracle
      `silu(group_norm(x, num_groups, weight, bias, eps))`.
    - Regression rows include `(2, 64, 32, 32)`, `(1, 64, 4, 16, 16)`,
      `(4, 128)`, and `(1, 128, 20, 256, 256)`, all with `num_groups=32`.
    - Regression dtypes include fp16, bf16, and fp32 with tolerances:
      fp16 `atol=3e-3`, `rtol=3e-3`; bf16 `atol=7e-2`, `rtol=2e-2`; fp32
      `atol=1e-5`, `rtol=1e-5`.
    - The wrapper-style `apply_group_norm_silu` path covers the 2D and 3D rows
      for fp16 and bf16.
    - Correctness checks shape, dtype, finite outputs, max absolute/relative
      error, and poisoned-output overwrite behavior.
  - Negative Tests (expected to FAIL):
    - Candidate output containing NaN/Inf, stale poisoned cells, wrong dtype,
      wrong shape, or tolerance violation fails correctness.
    - Invalid `num_groups`, channels not divisible by groups, unsupported rank,
      or unsupported dtype is rejected by validator tests rather than silently
      producing undefined output.

- AC-5: Benchmark method follows the standard standalone template.
  - Positive Tests (expected to PASS):
    - `bench/benchmark.py` starts from the shared standalone diffusion
      benchmark template and keeps the template policy unless a documented bug
      requires a fix and both sides are remeasured.
    - Timed regions exclude input generation, imports, JIT/build, Python setup,
      allocation, and data restoration.
    - Benchmark uses CUDA events as primary time, warmups, isolated runner when
      possible, fresh random inputs per trial, stable tensor objects inside a
      trial, deterministic interleaved A/B sampling, and inner-loop
      amplification up to the configured cap.
    - Results report median, mean, std, min, p10, p90, raw samples, per-workload
      speedup, equal-weight production geomean, and arithmetic mean.
  - Negative Tests (expected to FAIL):
    - A result without correctness passing first is invalid.
    - A result without provenance, GPU state, workload count, and settings is
      invalid.
    - Wrapper-inclusive wall clock as the only primary timing signal is
      invalid.

- AC-6: B200 execution and remote/container state are recorded.
  - Positive Tests (expected to PASS):
    - Later correctness, benchmark, profiling, and NCU runs execute on an
      NVIDIA B200.
    - `docs/run_log.md` records selected host, GPU id, GPU model, and
      before/after `nvidia-smi` state.
    - The same selected GPU is used consistently for baseline, candidate,
      correctness, benchmark, profiling, and NCU within a run.
    - Build logs, benchmark logs, profiler traces, and NCU reports are kept in
      a task-owned local or remote workspace.
  - Negative Tests (expected to FAIL):
    - H200 or non-B200 benchmark evidence cannot satisfy this B200 task.
    - Mixing different GPUs for baseline and candidate in the same final
      comparison invalidates the result.

- AC-7: KernelWiki and prior-art review inform the fresh search.
  - Positive Tests (expected to PASS):
    - Before selecting or revising a candidate strategy, the later Goal reads
      local `KernelWiki` skill guidance or `external/KernelWiki/SKILL.md` when
      available.
    - `docs/research.md`, `docs/draft.md`, or `docs/results.md` records
      relevant prior-art ideas, which ideas were adopted or rejected, and why.
    - Prior-art review is refreshed before each bounded optimization iteration
      when new benchmark or profiling evidence changes the decision.
  - Negative Tests (expected to FAIL):
    - A final report that contains an unexplained candidate design with no
      prior-art or design rationale fails this criterion.
    - Importing prior KDA candidate designs or conclusions as prior art is
      rejected.

- AC-8: Profiling and NCU evidence are tied to optimization decisions.
  - Positive Tests (expected to PASS):
    - After a correct candidate exists, if benchmark evidence shows a
      performance gap, plateau, regression, or unclear active bound, the later
      Goal reads local `ncu-report-skill` guidance or
      `external/ncu-report-skill/SKILL.md` when available.
    - NCU or profiler output is summarized into a bottleneck digest that names
      the active bound and the next edit it supports.
    - Final success or no-go includes roofline-style reasoning: estimated bytes
      moved, useful operations, achieved bandwidth and/or FLOP/s when relevant,
      and active bound/blocker.
  - Negative Tests (expected to FAIL):
    - Making repeated optimization edits after a plateau without new benchmark,
      profiler, NCU, or roofline evidence fails this criterion.
    - Keeping raw NCU/profiler artifacts as the only final explanation fails
      this criterion.

- AC-9: Iteration ledger is fresh, bounded, and evidence-backed.
  - Positive Tests (expected to PASS):
    - A fresh `docs/draft.md`, `docs/research.md`, or task-local
      `docs/solutions.jsonl`-style ledger records each attempt, source hash,
      changed files, correctness status, benchmark status, profiling status,
      promote/reject reason, and next action.
    - The search proceeds baseline -> first correct candidate -> benchmark ->
      profile/NCU when needed -> bounded edit -> promote/reject.
    - Shape specialization, template variants, autotune tables, or dispatchers
      are used only when evidence shows different shape buckets need different
      tradeoffs; `docs/dispatch.md` records bucket conditions and per-bucket
      evidence.
  - Negative Tests (expected to FAIL):
    - Reusing any KDA `solutions.jsonl`, candidate history, result summary, or
      optimization log is rejected.
    - Declaring no-go after the first losing candidate without baseline
      numbers, attempted candidate evidence, and named active blocker is
      rejected.

- AC-10: Final report is reproducible, complete, and honest.
  - Positive Tests (expected to PASS):
    - `docs/results.md` includes final commands, environment, baseline commit,
      candidate source hash, GPU host/id/model, workload count, correctness
      summary, per-workload performance table, production geomean speedup,
      arithmetic mean speedup, profiling/roofline explanation, conclusion, and
      residual risks.
    - If no absolute threshold exists in the source prompt, successful
      completion requires correctness plus relative improvement over the
      same-environment recovered baseline, measured by the frozen production
      geomean, with sufficient evidence.
    - If the Goal ends in no-go, `docs/results.md` separates what was proven,
      what was attempted, what blocked further progress, and what input or
      environment change would unlock more work.
  - Negative Tests (expected to FAIL):
    - Invented performance numbers, unmeasured claims, or claims copied from
      KDA run artifacts fail this criterion.
    - A final report without enough data to reproduce the final comparison
      fails this criterion.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The later Goal may fully recover the baseline, build a local benchmark harness,
implement one or more candidate kernels or dispatch buckets, run correctness,
benchmark, profiling, and NCU on B200, and produce all required task-local
evidence and reports. It may use task-owned remote scratch space for builds and
profiling artifacts when needed, while keeping final reproducibility notes in
this task's `docs/`.

### Lower Bound (Minimum Acceptable Scope)

The minimum valid completion includes fresh baseline recovery, local ABI for
baseline and candidate, complete frozen workloads, correctness harness passing
production and regression rows, final B200 benchmark against the same-environment
baseline, and `docs/results.md` showing either a correctness-preserving
production geomean speedup or an evidence-backed no-go. Anything less is not a
completed Goal.

### Allowed Reads

- This `plan.md`.
- Source task prompt:
  `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/kernels/b200_diffusion_group_norm_silu__multi_shape/prompt.md`.
- Shared public task rules explicitly referenced by the source prompt:
  `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/docs/standalone_diffusion_benchmark.md`,
  `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/docs/diffusion_kernel_rules.md`,
  `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/docs/diffusion_correctness_contract.md`,
  `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/docs/diffusion_benchmark_shape_coverage.md`,
  and
  `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/docs/standalone_diffusion_benchmark_template.py`.
- Codex Goal launcher docs and scripts under
  `/Users/bbuf/工作目录/Common/KDA-Pilot/codex-goal-diffusion/`.
- Upstream SGLang source needed to recover the target baseline and static
  interface/spec/test files clearly needed to recover the ABI or oracle, such
  as the canonical upstream `test_group_norm_silu.py` named by the correctness
  contract.
- Local `KernelWiki` and `ncu-report-skill` instructions, or
  `external/KernelWiki/SKILL.md` and `external/ncu-report-skill/SKILL.md` when
  present.

### Allowed Writes

- This task directory:
  `/Users/bbuf/工作目录/Common/KDA-Pilot/codex-goal-diffusion/b200_diffusion_group_norm_silu__multi_shape/`.
- Within that directory, expected writes are limited to `baseline/`,
  `solution/`, `bench/`, `docs/`, build metadata, and small task-local config
  needed by the harness.
- Task-owned remote scratch/workspace paths may be used for build outputs,
  benchmark logs, profiler traces, and NCU reports, provided final summaries
  and reproduction commands are copied into `docs/`.

### Forbidden Reads And Reuse

- Do not read, summarize, copy, or reuse any KDA optimization process or result
  under `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/kernels/**`, except the
  current source `prompt.md` and clearly static interface/spec files.
- Do not read or use existing KDA `baseline/`, `solution/`, `bench/`,
  task-local `docs/`, `src/`, `tests/`, `profile/`, `ncu/`,
  `benchmark.csv`, `solutions.jsonl`, logs, captured tensors, profiler traces,
  NCU reports, draft summaries, result summaries, candidate code, or run notes.
- Do not treat any prior KDA run performance number, conclusion, candidate
  design, failure path, or success path as an input fact.

### Forbidden Writes And Pollution

- Do not modify source KDA task directories under `diffusion/kernels/`.
- Do not modify shared diffusion docs or upstream SGLang source as part of this
  task.
- Do not write artifacts into another Codex Goal task directory.
- Do not leave raw profiler traces, raw NCU reports, temporary build products,
  large scratch logs, or failed experiment dumps staged for PR-like output.

### Upstream Integration

Upstream integration is out of scope unless the user explicitly asks later.
This Goal's completion surface is the standalone task-local artifact set and
evidence report.

## Baseline Recovery Strategy

1. Resolve the latest upstream SGLang `main` commit at baseline-recovery time.
2. Record in `docs/baseline_source.md`:
   upstream URL, branch, resolved commit SHA, resolution timestamp, copied
   files, source entry points, dependency versions, local ABI, build method,
   compile flags, and any local wrapper decisions.
3. Copy only the source needed for:
   `triton_group_norm_silu` and `apply_group_norm_silu` into `baseline/`.
4. Recover public callable arguments, tensor layouts, scalar parameters, return
   or output semantics, dtype behavior, and stream behavior from upstream
   source and static tests/specs.
5. Expose the copied baseline through local low-overhead ABI entry points.
   Prefer `TVM_FFI_DLL_EXPORT_TYPED_FUNC`, `tvm::ffi::TensorView` arguments,
   output tensors passed last, and destination passing style when using CUDA.
6. If the copied SGLang baseline is Triton or Python, keep it task-local and
   build a local adapter with the same benchmark ABI used by the candidate.
7. Do not begin candidate optimization until baseline correctness and the
   first benchmark harness path are working locally.

## Correctness Plan

- Use the independent PyTorch/math oracle:
  `silu(group_norm(x, num_groups, weight, bias, eps))`.
- Implement oracle rows for the canonical regression grid:
  2D image, 3D video, token, and large-tile shapes listed in AC-4.
- Include production workload correctness for every row in
  `bench/workloads.json`.
- Use fp16, bf16, and fp32 regression dtype coverage with contract tolerances.
- Use fp16 production dtype for HunyuanVideo VAE rows unless recovered workload
  provenance says otherwise.
- Check candidate against both independent oracle rows and copied baseline
  rows. If a full oracle is too expensive for every production row, use full
  oracle coverage on regression rows plus sampled production rows, and still
  compare all production rows against the copied local baseline.
- Poison outputs before every correctness run.
- Verify output shape, dtype, finite values, max absolute error, max relative
  error, and that all output cells are written.
- Add negative tests for invalid group/channel divisibility, unsupported rank,
  unsupported dtype, missing weight/bias where required, and non-contiguous
  layouts not claimed as supported.
- Correctness must pass before any benchmark result is considered valid.

## Benchmark Plan

- Populate `bench/workloads.json` before tuning. It is the workload source of
  truth and must stay frozen through a benchmark series.
- Prefer a fresh live HunyuanVideo capture for this many-row family. If using a
  retained raw shape-capture source mentioned by the shared shape coverage doc,
  first verify it is shape coverage rather than a task-local optimization
  result, record provenance, and do not use any KDA candidate/result data.
- Copy the standard benchmark template to `bench/benchmark.py` and keep its
  policy intact.
- Implement `bench/adapter.py` with only:
  `make_case(workload, *, device, seed)`,
  `call_baseline(workload, inputs, outputs)`, and
  `call_candidate(workload, inputs, outputs)`, plus optional
  `compare_outputs`.
- Use recommended defaults unless `docs/benchmark_method.md` records a reason:
  warmup 10, 7 trials, inner iterations min 1, max 4096, target sample about
  1000 us, timeout 600 s, isolated runner.
- Generate fresh random inputs per trial while keeping tensors stable inside a
  trial.
- Preallocate outputs outside timing.
- Use CUDA events as primary GPU timing and wrapper-inclusive wall time only as
  diagnostics.
- Use deterministic interleaved A/B sampling to reduce drift.
- Report per workload:
  median, mean, std, min, p10, p90, raw samples, baseline wall samples,
  candidate wall samples, and speedup.
- Report headline:
  equal-weight geometric mean over production workloads, arithmetic mean,
  min/max speedup, passed/total, and correctness summary.
- If methodology, workload, tolerance, or scoring changes after tuning begins,
  delete or quarantine stale results and remeasure both baseline and candidate.

## GPU, Container, And Remote Constraints

- The later Goal must run correctness, benchmark, profiling, and NCU evidence
  for final claims on NVIDIA B200.
- Before GPU work, inspect `nvidia-smi` and choose a GPU with no active compute
  process and no meaningful memory occupancy.
- Use the same GPU for baseline, candidate, correctness, benchmark, profiling,
  and NCU within a final comparison.
- Record host, container if any, GPU id, GPU model, driver/CUDA/PyTorch/compiler
  versions, and before/after GPU state in `docs/run_log.md` and final
  benchmark provenance.
- Use task-owned remote workspace paths for temporary execution artifacts.
- This plan-generation turn must not run benchmark, profiler, NCU, or remote
  machine commands. Those are later Goal actions only.

## KernelWiki Prior-Art Plan

- Before choosing the first nontrivial candidate and before each optimization
  iteration that could change the design, read local `KernelWiki` skill
  guidance or `external/KernelWiki/SKILL.md` when available.
- Search for Blackwell/B200-relevant evidence about reductions,
  normalization-like kernels, memory bandwidth, vectorized loads/stores,
  occupancy/register tradeoffs, warp/block reductions, and shape-specialized
  dispatch.
- Inspect relevant upstream source or docs only when they could change the
  design: SGLang, CUDA/CuTe/Triton examples, PyTorch, vLLM, TensorRT-LLM,
  FlashInfer, or other kernel references.
- Record each adopted or rejected idea in `docs/research.md`, `docs/draft.md`,
  or `docs/results.md`, including the evidence used and why it applies or does
  not apply to GroupNorm+SiLU production shapes.
- Do not import candidate designs or conclusions from prior KDA runs.

## ncu-report-skill And Profiling Plan

- Read local `ncu-report-skill` guidance or
  `external/ncu-report-skill/SKILL.md` once a correct candidate exists and any
  of these are true:
  performance regresses, production geomean does not improve, a shape bucket is
  unclear, a plateau appears, or final evidence needs an active-bound
  explanation.
- Profile focused representative workloads and any outlier buckets rather than
  only the easiest row.
- Keep raw NCU/profiler artifacts in task-owned scratch or task-local
  non-final directories, and summarize the actionable evidence in docs.
- Convert NCU evidence into a bottleneck digest:
  relevant metrics, memory traffic estimate, achieved bandwidth, occupancy or
  stall signals, instruction mix if useful, active bound, and the next edit
  chosen from that evidence.
- Bind every post-profile optimization edit to the profile evidence that
  motivated it.
- Final success or no-go must include a roofline-style explanation, even when
  full NCU collection is blocked; if blocked, document why and use the best
  available benchmark-derived estimate.

## Iteration Strategy

1. Re-read this plan, source prompt, and shared rules.
2. Recover upstream baseline and local ABI.
3. Generate and freeze `bench/workloads.json`; audit current presets.
4. Build `bench/correctness.py`, `bench/adapter.py`, and template-derived
   `bench/benchmark.py`.
5. Validate baseline correctness and harness behavior.
6. Establish baseline benchmark evidence on B200.
7. Review KernelWiki/prior art and write the first candidate design rationale.
8. Implement the smallest first candidate that can pass correctness.
9. Run correctness. If it fails, fix or reject the attempt with evidence.
10. Run benchmark. Promote only if evidence improves the frozen production
    geomean without correctness regression.
11. If benchmark evidence is inconclusive or negative, profile representative
    rows and choose one next edit from the evidence.
12. Maintain a fresh search ledger in `docs/draft.md`, `docs/research.md`, or
    `docs/solutions.jsonl` with attempt id, source hash, changed files,
    workloads touched, correctness result, benchmark result, profiling result,
    promote/reject reason, and next action.
13. If using shape buckets or dispatch, document bucket predicates and
    per-bucket evidence in `docs/dispatch.md`.
14. Stop on success only after final correctness, final benchmark, final docs,
    and final evidence checks pass.
15. Stop on no-go only after baseline numbers, at least one reasoned candidate,
    correctness status, benchmark evidence, profiling or roofline evidence, and
    a named blocker are documented.

## Dependencies And Sequence

1. Baseline and ABI milestone:
   recover source, provenance, local baseline calls, and candidate ABI skeleton.
2. Workload and correctness milestone:
   build workloads, preset audit, oracle, regression grid, negative tests, and
   correctness gate.
3. Benchmark milestone:
   template-based benchmark, frozen method docs, baseline benchmark, candidate
   benchmark, provenance.
4. Research and profiling milestone:
   KernelWiki review, first candidate rationale, NCU/profiling when needed,
   roofline-style analysis.
5. Optimization milestone:
   bounded search loop, promote/reject ledger, optional dispatch buckets.
6. Final report milestone:
   final correctness, final benchmark, per-shape comparison, conclusion,
   no-go or success status, and reproduction commands.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Re-read plan, source prompt, shared docs, and establish greenfield boundaries | AC-1, AC-9 | analyze | - |
| task2 | Recover latest upstream SGLang baseline and provenance | AC-1 | coding | task1 |
| task3 | Build matching local baseline/candidate ABI skeleton | AC-2 | coding | task2 |
| task4 | Generate complete workloads and preset audit | AC-3 | coding | task2 |
| task5 | Implement correctness oracle, regression grid, and negative tests | AC-4 | coding | task3, task4 |
| task6 | Copy and adapt standard benchmark template and adapter | AC-5 | coding | task3, task4 |
| task7 | Run B200 baseline correctness and benchmark with provenance | AC-5, AC-6 | analyze | task5, task6 |
| task8 | Review KernelWiki/prior art and write first candidate rationale | AC-7 | analyze | task7 |
| task9 | Implement first correct candidate | AC-2, AC-4 | coding | task8 |
| task10 | Benchmark candidate and update fresh attempt ledger | AC-5, AC-9 | analyze | task9 |
| task11 | Run profiling/NCU or roofline analysis when needed | AC-8 | analyze | task10 |
| task12 | Iterate bounded candidate edits or dispatch buckets | AC-8, AC-9 | coding | task10, task11 |
| task13 | Write final results, success/no-go conclusion, and reproduction notes | AC-10 | coding | task10 |

## Open Questions, Defaults, And Blockers

- Default: no absolute latency target exists in the source prompt, so the
  success bar is correctness plus relative improvement over the same-environment
  recovered baseline, measured by frozen production geomean and backed by
  reproducible evidence.
- Default: live HunyuanVideo capture is preferred for workload generation.
  Retained raw shape-capture data may be used only if it is clearly permitted
  shape coverage and not a task-local KDA optimization artifact.
- Default: no upstream integration or PR packaging is required.
- Default: final raw profiler/NCU artifacts remain local or scratch; final docs
  contain summaries and commands.
- Blocker: no accessible B200 means final benchmark claims cannot complete.
- Blocker: inability to recover upstream baseline ABI means stop with
  provenance of attempted source recovery and the missing interface detail.
- Blocker: incomplete workload coverage without live capture or permitted
  retained shape coverage means stop before optimization.
- Blocker: correctness failures that cannot be isolated after bounded attempts
  mean no benchmark result may be claimed.
- No-go condition: after a recovered baseline, complete workloads, at least one
  reasoned correct candidate, benchmark evidence, and profiling/roofline
  evidence, no candidate path shows a defensible production geomean speedup and
  the active blocker is documented.

## Implementation Notes

- Code and comments created by the later Goal must use domain names, not plan
  terminology such as `AC-`, `Milestone`, or `Phase`.
- Keep edits scoped to this task directory.
- Prefer simple, auditable harness code over clever benchmark infrastructure.
- Do not invent performance numbers. Every performance claim must point to a
  recorded command, workload file, environment, and result artifact.
- Do not mark the Codex Goal complete because the attempt is merely plausible;
  mark it complete only after the evidence in this plan is satisfied.
