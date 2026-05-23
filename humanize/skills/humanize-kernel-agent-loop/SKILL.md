---
name: humanize-kernel-agent-loop
description: "Run an autonomous Humanize Kernel Agent Loop for GPU kernel optimization: plan/refine K/R/W into task-acceptance pairs, create a clean standalone repo, research with KernelWiki, iterate with benchmark/profile evidence, autotune across the workload distribution, emit kernels/dispatcher/tuning decisions, maintain ledgers, and start RLCR."
type: flow
---

# Humanize Kernel Agent Loop

Use this flow when the user wants an autonomous kernel optimization loop, not a
general software feature loop. This is a kernel-specialized wrapper around
Humanize RLCR.

## Algorithm Mapping

This skill follows the paper-style Humanize kernel loop architecture in a
KernelPilot standalone-repo variant. The input contract is:

```text
Require: kernel definition K, correctness reference R, workload distribution W
Ensure: kernel implementation K_hat that passes correctness checks on W and
        minimizes latency, plus dispatcher and tuning decisions when W contains
        multiple regimes.
```

The loop has three stages that run in order: Stage 1 closes with a research
digest before Stage 2's first lineage, and Stage 2 closes when the reviewer
accepts the correctness/benchmark evidence before Stage 3 starts the autotune
pass.

- Research: inspect the target kernel contract, baseline/reference behavior,
  workload distribution, repository context, and `KernelWiki` evidence.
  Extract implementation recipes, profile hypotheses, and benchmark priorities.
- Iterate: write candidate kernels in the standalone repo, compile, test,
  benchmark, profile with `ncu-report-skill`/Nsight Compute when useful, query
  `KernelWiki` when prior art or hardware evidence can guide the next
  edit, and record every lineage decision.
- Autotune: scan the target workload distribution, benchmark selected variants
  or configurations, build a performance map, and emit a shape-aware dispatcher
  instead of forcing all shapes through one monolithic kernel.

Planning decomposes the work into task-acceptance pairs `P = {(t_i, ac_i)}`.
Each pair is executed until the review gate accepts the evidence:

```text
for each (t_i, ac_i) in P:
    done = false
    while not done:
        writer executes t_i in the standalone repo
        # inspect/edit code, compile, test, benchmark, profile, query evidence
        verdict, feedback = reviewer checks evidence against ac_i
        if verdict == pass:
            done = true
        else:
            writer refines the kernel using feedback
            # if blocked, consult reviewer/tool evidence; ask human only if still unresolved
return final kernels, dispatcher, and tuning decisions
```

In diagrams or papers that say "Claude executes, Codex reviews", read "Claude"
as "the current writer agent". In a Claude Code session that may be Claude; in
a Codex session that may be Codex. The review model is controlled by Humanize
configuration or CLI flags, not by this skill.

The reviewer in the diagram is the Humanize Stop hook. After each round the
hook runs the configured review against the round's evidence and either accepts
the round (pass) or emits a next-round prompt (blocked feedback) that the
writer follows verbatim before the next round.

The installer hydrates these paths:

```text
Humanize runtime: {{HUMANIZE_RUNTIME_ROOT}}
KernelPilot root: {{KERNELPILOT_ROOT}}
KernelWiki root: {{KERNELWIKI_ROOT}}
ncu-report-skill root: {{NCU_REPORT_SKILL_ROOT}}
```

If `{{KERNELWIKI_ROOT}}` or `{{NCU_REPORT_SKILL_ROOT}}` was not hydrated,
locate the installed sibling skills named `KernelWiki` and
`ncu-report-skill`, or use the KernelPilot checkout defaults under
`external/KernelWiki` and `external/ncu-report-skill`.

## Contract

Run the Humanize steps inside this skill. Do not ask the user to run separate
`gen-plan`, `refine-plan`, or `humanize-rlcr` commands.

Given a kernel task and target, you must recover or define `K`, `R`, and `W`,
then:

1. Build the kernel-specific task-acceptance plan yourself.
2. Refine it yourself.
3. Create or enter a clean standalone optimization repo.
4. Record the workload distribution `W` and benchmark/correctness contract.
5. Encode Stage 1 research, Stage 2 iteration, and Stage 3 autotuning as
   task-acceptance pairs in the refined plan.
6. Start Humanize RLCR on the refined plan.
7. Execute the current round until the Stop hook takes over. The current and
   later rounds run the research, implementation, profiling, benchmark,
   autotuning, dispatcher, and review tasks until accepted.

Ask the user only if the kernel definition `K`, correctness reference `R`,
workload distribution `W`, target GPU, comparison target, or hard scope
constraint is missing and cannot be inferred safely.
After those inputs exist, do not ask the user to approve each implementation
strategy, knowledge query, profiling run, benchmark expansion, or lineage reset.
The loop owns those tactical decisions and records the evidence behind them.

## Loop Defaults

- Candidate implementation language is autonomous unless the user explicitly
  fixes it. If the user specifies CUDA C++/PTX, Triton, CuTe DSL, TileLang,
  CUTLASS/CuTe, ThunderKittens, torch.compile/Inductor, or another kernel stack,
  use that stack.
- If the user does not specify a language, choose and revise the implementation
  system most likely to reach the stated correctness and performance target
  from local context, available baselines, prior art, benchmarks, and profiler
  evidence.
- Treat `W` as a workload distribution, not just a smoke test. If the user gives
  explicit benchmark cases, those cases define `W`. If the user gives a model,
  trace, or serving scenario, derive representative cases and record how they
  were sampled. If only one focused shape exists, record that `W` is a single
  regime and make the dispatcher/tuning decision trivial.
- Treat `KernelWiki` and `ncu-report-skill`/Nsight Compute as autonomous tools
  for deciding what to try next. The user should not need to tell the loop when
  to research, profile, or switch implementation strategy.
- Treat "standalone" as a repository, build, benchmark, and runtime boundary.
  The candidate implementation, tests, benchmark harness, profile artifacts,
  and ledgers live in the isolated optimization repo.
- If external source directly affects code, record the exact source path/URL,
  commit or version, license/notice, copied/adapted files, and optimization
  delta in the attempt ledger or lineage.
- Keep the source framework checkout itself read-only for standalone work unless
  the user explicitly asks for an in-place framework patch.
- Run optimization work in a fresh standalone git repo with its own PyTorch
  binding, correctness tests, benchmark harness, ledgers, lineage, and profile
  artifacts.
- Every correct candidate attempt gets an attempt-ledger row. Only correct
  candidates that improve performance get an optimization-ledger row.
- Stage 1 research must leave a concise recipe/evidence artifact before the
  first serious optimization lineage. It should record source/baseline findings,
  candidate implementation routes, expected bottlenecks, and initial benchmark
  priorities. It is not a proof-of-reading ledger; it is the first decision map.
- Prefer collecting one baseline `ncu-report-skill` digest for a representative case
  after baseline correctness/benchmark succeeds. Skip it when Nsight Compute is
  unavailable or when a cheaper measurement is sufficient for the current round,
  and record the reason.
- Use `ncu-report-skill` again whenever profiler evidence is the best available way
  to explain a regression, plateau, surprising win, bottleneck shift, or next
  implementation edit.
- The loop remains incomplete while correctness, benchmark, provenance, lineage,
  workload coverage, dispatcher/tuning, or evidence gaps prevent a trustworthy
  autonomous next decision or final claim.

## Required Files In The Standalone Repo

Create these before starting RLCR, then keep them updated during the loop:

```text
.gitignore
.humanize/kernel-agent/refined-plan.md
README.md
workloads/
src/<task_name>/
bindings/
tests/
benchmarks/
dispatch/
ledgers/attempt-ledger.md
ledgers/optimization-ledger.md
ledgers/lineage.jsonl
ledgers/research-digest.md
ledgers/tuning-decisions.md
benchmarks/performance-map.json
profile-artifacts/README.md
```

Before the first scaffold commit, `.gitignore` must protect local Humanize loop
state:

```gitignore
.humanize*
```

The refined plan file must remain present on disk but untracked unless the loop
is intentionally started with `--track-plan-file`. For the default startup
command below, verify this before RLCR setup:

```bash
git check-ignore .humanize/kernel-agent/refined-plan.md
if git ls-files --error-unmatch .humanize/kernel-agent/refined-plan.md >/dev/null 2>&1; then
  git rm --cached .humanize/kernel-agent/refined-plan.md
fi
```

Commit `.gitignore` and the standalone scaffold, not `.humanize/` loop state.
This matches the `setup-rlcr-loop.sh` gate: without `--track-plan-file`, a
tracked `.humanize/kernel-agent/refined-plan.md` is rejected.

## Knowledge Evidence

The installed sibling skill `KernelWiki` carries the full protocol; this
section is the loop-specific summary. Stage 1 findings land in
`ledgers/research-digest.md`, and any code-level borrowing shows up in the
matching attempt row, lineage entry, or profile digest.

Three peer routes are available. They cover non-overlapping evidence:

- **Route A — KernelWiki PR pages and artifacts.** Materialized merged-PR
  pages plus fetched `diff.patch` / provenance bundles from the major kernel
  frameworks (SGLang, vLLM, TensorRT-LLM, PyTorch, FlashAttention, FlashInfer,
  CUTLASS/CuTe, CCCL, Triton, DeepGEMM, ThunderKittens, TileLang, QuACK,
  DeepSeek TileKernels).
- **Route B — KernelWiki synthesis, blogs, docs, and contest pages.** Wiki
  synthesis pages, extracted blog/code-source notes, official doc summaries,
  competition pages, and query indices. These are decision aids; trace any
  code-level borrowing back to their listed PR/source IDs.
- **Route C — Live web / official / upstream.** Web search, official docs,
  GitHub PR pages, and upstream source consulted online.

### Stage 1 Three-Route Sweep (Required)

Before plan refinement closes, run all three routes to build the research
digest. Combining all three is how the plan picks an implementation route,
expected bottlenecks, and benchmark/profile priorities. Each route gets at
least one citation in `ledgers/research-digest.md`, or an explicit "no relevant
material" note when the route returned nothing.

### Later Stages (Agent-Driven)

After Stage 1, the loop owns the call on when to query, which routes to use,
and how deep to go. Use the routes whenever evidence helps the current
implementation choice, benchmark result, profile digest, plateau/regression
explanation, reviewer question, or next kernel edit.

Run KernelWiki scripts from `{{KERNELWIKI_ROOT}}`, the directory that contains
`KernelWiki`'s `SKILL.md`.

### Route A: Local PR Diffs

```bash
cd {{KERNELWIKI_ROOT}}
python3 scripts/query.py "tcgen05" --architecture B200 --limit 10
python3 scripts/query.py --repo pytorch/pytorch --tag tma --compact
python3 scripts/grep_wiki.py "tcgen05|tmem" --only prs
python3 scripts/get_page.py pr-pytorch-157241
less artifacts/prs/<repo-id>/PR-<number>/diff.patch
cat artifacts/prs/<repo-id>/PR-<number>/PROVENANCE.yaml
```

`query.py` filters by `--type`, `--tag`, `--repo`, `--language`,
`--architecture`, `--kernel-type`, `--symptom`, and `--confidence`. Combine
filters with the current kernel context. Open the page named by `id`, then
inspect the bundle named by `artifact_dir` (`diff.patch`, provenance, and any
captured derived files) before borrowing an idea.

### Route B: KernelWiki Synthesis, Blogs, Docs, And Contests

Use synthesis pages and indices to connect the current kernel problem to
patterns, upstream PRs, docs, blog code references, and contest solutions.
These pages are not a substitute for source-level review; follow each useful
source ID back to Route A artifacts or Route C upstream code before adapting
implementation details.

```bash
cd {{KERNELWIKI_ROOT}}
python3 scripts/query.py "tma swizzle transpose" --type technique --limit 10
python3 scripts/grep_wiki.py "block-scaled|stream-k|topk" --only wiki --any
python3 scripts/get_page.py kernel-flash-attention-sm100-mla-topk --follow-sources
less queries/by-technique.md
```

Keep the current operator, dtype, architecture, and framework in the search
terms so the matches stay relevant.

### Route C: Live Web / Official / Upstream

Use live web search, official docs, GitHub PR pages, and current upstream
source as a peer route. When implementation details matter, prefer official
docs and upstream source over blogs or snippets. Record URLs, commit SHAs,
source paths, and license/notice text whenever an external-route finding
directly shapes the kernel.

### Shared Example

For `FlashAttention SM100 MLA TopK`:

- Route A surfaces `pr-flash-attention-2441` via `query.py`; its PR page and
  `artifacts/prs/flash-attention/PR-2441/diff.patch` carry the implementation
  evidence.
- Route B opens `kernel-flash-attention-sm100-mla-topk` and related technique
  pages to separate top-k gather, tile scheduling, and TMA considerations.
- Route C reads the upstream FlashAttention PR/page, current upstream source,
  and architecture-level docs.

### Citation Checklist

Apply the same shape to every finding before letting it shape code or reviews:

- Name the route(s) used.
- **Route A:** PR page ID, page path under `sources/prs/`, `artifact_dir`, and
  the specific artifact/provenance files cited.
- **Route B:** KernelWiki page IDs or query-index paths, their source IDs, and
  which source IDs were followed into Route A or Route C.
- **Route C:** URLs, commit SHAs or version tags, source paths, and any
  license/notice text required when the code is reused.
- If a route returns a thin or empty match for a detail that matters, widen
  the search inside that route or cross-check against another route before
  treating it as a finding.

KernelWiki synthesis pages, blogs, docs, contests, and query indices can guide
the search. Code-level claims still need a concrete source page, artifact, or
live upstream source before they shape implementation.

## Plan Requirements

Write `.humanize/kernel-agent/refined-plan.md` in the standalone repo. It must
use the Humanize gen-plan schema and include these acceptance criteria:

- Clean standalone repo exists and is committed before RLCR starts.
- Baseline framework checkout is protected from accidental edits unless the
  user asks for an in-place patch.
- Candidate implementation language is documented and follows the user request
  when fixed, otherwise the autonomous selected strategy.
- If external source seeds any candidate, provenance, license/notice,
  copied/adapted files, and the first optimization delta are recorded before
  further mutation.
- `K`, `R`, and `W` are explicit: the kernel semantics, correctness oracle,
  target workload distribution, target GPU, comparison baseline, and hard scope
  exclusions are recorded before the first candidate is selected.
- Correctness tests cover `W`, representative edge cases, dtype/layout/mode
  boundaries, and baseline/reference parity.
- Benchmark harness records per-shape timing, geomean, best/worst cases,
  workload coverage, and environment metadata.
- Stage 1 research digest records baseline and reference inspection: how the
  reference implementation lays out memory, launches kernels, handles
  dispatch/dtype/layout branches, and where its hot path lives.
- Stage 1 runs all three knowledge routes (A: KernelWiki PR pages/artifacts,
  B: KernelWiki synthesis/blog/doc/contest pages, C: live web/official/upstream).
  Each route gets at least one citation in `ledgers/research-digest.md`, or an
  explicit "no relevant material" note when the route returned nothing.
- Stage 1 research digest records baseline/source findings, evidence from the
  three routes that materially affects the plan, candidate implementation
  routes, suspected bottlenecks, and first benchmark/profile priorities.
- A baseline profile decision is recorded after baseline benchmark succeeds:
  either capture a representative `ncu-report-skill` digest or explain why the loop is
  using cheaper evidence first.
- Later `ncu-report-skill` profiles are captured autonomously when they are the best
  tool for regressions, plateaus, surprising wins, bottleneck shifts,
  profile-driven edits, or reviewer-requested evidence, then converted into NCU
  Report Digests.
- Attempt ledger records every version, including failed correctness,
  regressions, plateaus, and abandoned ideas.
- Optimization ledger records only correct versions with measured improvement.
- Lineage records parent version, mutation/motivation, influential source or PR
  evidence when it directly affects code, result, and selected/rejected status.
- Stage 3 autotuning builds a performance map over `W`, tests selected variants
  or configurations across all required regimes, and emits dispatcher/tuning
  decisions. If `W` is a single focused case, record why a trivial dispatcher is
  sufficient.
- The final answer and ledgers identify final kernels, fallback paths if any,
  dispatcher policy, tuning decisions, correctness matrix, benchmark matrix, and
  remaining unsupported regimes.
- When progress stalls, expand evidence research across the selected peer
  routes: unread PR pages/artifacts, KernelWiki synthesis pages, official docs,
  GitHub PR pages, linked tests, benchmarks, and profiler notes, guided by the
  current task context and existing attempt/lineage notes.
- When profiling shows a candidate is far below the target or in a different
  bottleneck class than the baseline, use the profile evidence to reassess the
  lineage. A useful next round either tests a concrete profile-driven edit,
  creates a smaller executable milestone, or records new evidence for
  continuing the current route.
- Experimental GPU kernels that can hang should use timeout-bounded bring-up
  milestones. A timeout marks that lineage rejected until a minimal executable
  tile validates the suspected protocol, descriptor, layout, memory, and
  synchronization semantics.
- Stop only when all ACs are met and the loop has either reached the stated
  target or recorded evidence that further autonomous optimization is no longer
  justified under the current scope, budget, and available implementation
  routes.

## Progress And Timeout Checks

- If an attempt times out or hangs, reproduce the failure with the smallest
  shape/tile under a hard timeout before any target-size benchmark. Record the
  rejected lineage and the suspected root-cause surface.
- If repeated reviews identify the same mainline blocker, narrow the next
  round to an executable milestone or design reset that can falsify the current
  hypothesis.
- If a candidate remains orders of magnitude below the target after a correct
  tensor-core-class attempt, treat further same-lineage tuning as evidence
  maintenance unless `ncu-report-skill` names a specific low-risk edit with plausible
  order-of-magnitude impact.

## NCU Report Evidence

Invoke the `ncu-report-skill` skill autonomously when profiler evidence is the best
next source of truth. These are heuristics, not user-facing gates:

- Baseline benchmark has passed and no baseline profile decision or NCU Report
  Digest exists.
- A correct candidate is within +/-2% of baseline across configured cases.
- A correct candidate regresses on one or more configured cases.
- The benchmark has plateaued or weakly improved and the next edit is unclear.
- A candidate is much faster than expected and needs explanation.
- A reviewer asks for an NCU Report Digest.

Persist `.ncu-rep`, raw CSV, source export, PM-sampling export when available,
and comparison paths in the digest. Each digest must end with one concrete next
kernel edit.

## RLCR Startup

After writing and committing the standalone repo scaffolding, start the loop
from inside the standalone repo:

```bash
"{{HUMANIZE_RUNTIME_ROOT}}/scripts/setup-rlcr-loop.sh" .humanize/kernel-agent/refined-plan.md --yolo
```

If setup exits non-zero, stop and report the error. Do not bypass the gate.
The loop uses Humanize's configured review model and default max iteration
limit unless the caller explicitly passes overrides such as `--max` or a model
flag. The current default max iteration limit is 84 rounds.

After setup succeeds:

1. Read `.humanize/rlcr/<timestamp>/round-0-prompt.md`.
2. Execute the current round.
3. Commit changes.
4. Write the required round summary.
5. Stop normally so the native Humanize Stop hook can review.

If the hook blocks exit, follow the generated next-round prompt exactly.
