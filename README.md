<div align="center">

# KernelPilot

**An autonomous Humanize-powered GPU kernel optimization loop wired to
KernelWiki evidence and the external Nsight Compute profiling skill.**

[![GitHub stars](https://img.shields.io/github/stars/BBuf/kernel-pilot?style=social)](https://github.com/BBuf/kernel-pilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/BBuf/kernel-pilot?style=social)](https://github.com/BBuf/kernel-pilot/forks)
[![Last commit](https://img.shields.io/github/last-commit/BBuf/kernel-pilot?style=flat-square)](https://github.com/BBuf/kernel-pilot/commits/main)
[![KernelWiki PR pages](https://img.shields.io/badge/KernelWiki_PR_pages-2692-2ea44f?style=flat-square)](external/KernelWiki/sources/prs/)
[![Knowledge cutoff](https://img.shields.io/badge/cutoff-2026--05--20-8250df?style=flat-square)](external/KernelWiki/data/refresh-cutoff.yaml)

</div>

KernelPilot is for serious CUDA kernel tuning runs where the important facts
are easy to lose: which upstream PR inspired a candidate, which shape regressed,
what Nsight Compute actually said, which evidence changed the next edit, and
whether the candidate belongs in a framework repo or a clean experiment.

The project now keeps Humanize loop logic in this repository and uses two
external skill submodules for kernel evidence and profiling:

| Skill | Source | Role |
| --- | --- | --- |
| [`humanize-kernel-agent-loop`](humanize/skills/humanize-kernel-agent-loop/) | KernelPilot | Creates a clean kernel optimization loop around `K`, `R`, and `W`: standalone workspace, correctness/benchmark evidence, lightweight ledgers, optional profiling, optional prior-art lookup, and Humanize review. |
| [`KernelWiki`](external/KernelWiki/) | [`BBuf/KernelWiki`](https://github.com/BBuf/KernelWiki/tree/kernelpilot-knowledge-expansion) | Use when prior PRs, wiki synthesis, docs, blogs, or upstream kernel examples can inform the current design. |
| [`ncu-report-skill`](external/ncu-report-skill/) | [`DongyunZou/ncu-report-skill`](https://github.com/DongyunZou/ncu-report-skill) | Use when Nsight Compute evidence is needed to profile a kernel, diagnose a bottleneck, explain a regression, or choose the next optimization edit. |

Together they make an optimization loop that can work from a simple request:

```text
[$humanize-kernel-agent-loop] Optimize SGLang's GEMM path for M=64, N=2048, K=2048, fp16, bias=true, and beat the current SGLang baseline by at least 10%.
```

The loop keeps the engineering structure in place while staying light on
policy: set up a clean workspace, prove correctness, measure latency, consult
KernelWiki when prior art helps, profile when profiling would change the next
edit, and let Humanize review the round evidence.

## Why Use It

- **Prior art when useful.** The agent can use KernelWiki PR artifacts,
  synthesis pages, blogs/docs/contests, and live upstream research when they
  help the current kernel decision.
- **Standalone by default.** Candidate kernels do not pollute SGLang, vLLM,
  PyTorch, or other large framework repos. The loop creates an isolated repo
  with bindings, tests, benchmarks, ledgers, lineage, and profile artifacts.
- **Profiling when needed.** The agent uses `ncu-report-skill` when Nsight
  Compute evidence can explain a baseline, regression, plateau, surprising win,
  or next optimization edit.
- **Review-gated iteration.** Humanize RLCR keeps the loop from declaring
  victory too early; default loop budget is 84 iterations unless configured
  otherwise.
- **Shape-aware tuning.** The loop treats benchmark cases as a workload
  distribution, builds a performance map, and emits dispatcher/tuning decisions
  when different regimes need different kernels or configurations.

## Kernel Agent Loop

<p align="center">
  <img src="https://raw.githubusercontent.com/BBuf/kernel-pilot/main/docs/assets/kernel-agent-loop.svg" alt="KernelPilot Agent Loop" width="620">
</p>

## KernelWiki

KernelWiki lives in [`external/KernelWiki`](external/KernelWiki/). KernelPilot
points this submodule at the `kernelpilot-knowledge-expansion` branch of
`BBuf/KernelWiki`, which includes the KernelPilot-only PR and source knowledge
merged through KernelWiki's own candidate ledgers, PR page generator, artifact
fetcher, index generator, and validator.

Current snapshot:

| Corpus layer | Contents |
| --- | --- |
| PR pages | 2,692 merged CUDA/Triton/CuTe/CUTLASS-related PR pages from SGLang, vLLM, TensorRT-LLM, PyTorch, FlashAttention, FlashInfer, CUTLASS/CuTe, CCCL, Triton, DeepGEMM, ThunderKittens, TileLang, QuACK, and DeepSeek TileKernels. |
| PR artifacts | Fetched `diff.patch` and provenance bundles for selected high-value PRs, including FlashAttention SM100 MLA TopK, TensorRT-LLM Blackwell DSA/indexer, and CCCL scan/memory primitives. |
| Synthesis | KernelWiki wiki pages, blog/code-source notes, docs, contests, and generated query indices for hardware features, techniques, repos, languages, and kernel types. |

Query examples:

```bash
cd external/KernelWiki
python3 scripts/query.py "TensorRT FP4 DSA indexer" --limit 5 --compact
python3 scripts/query.py "FlashAttention SM100 MLA topk" --limit 5 --compact
python3 scripts/query.py --repo sglang --tag tma --compact
python3 scripts/grep_wiki.py "tcgen05|tmem" --only prs
python3 scripts/get_page.py kernel-flash-attention-sm100-mla-topk --follow-sources
python3 scripts/validate.py
```

## ncu-report-skill

`ncu-report-skill` lives in
[`external/ncu-report-skill`](external/ncu-report-skill/). It provides the
profiling workflow used by the loop when profiler evidence is the best next
source of truth:

```bash
cd external/ncu-report-skill
less reference/01-workflow.md
less reference/03-collection.md
less reference/08-b200-metric-names.md
```

The skill emphasizes one run directory per profile, standalone harnesses when
possible, `ncu --set full` plus source-level collection, Python parsing through
the `ncu_report` API, and a final report that ranks evidence-backed next edits.

## Install

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/BBuf/kernel-pilot.git
cd kernel-pilot
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

### Claude Code Install

```bash
humanize/scripts/install-skills-claude.sh
```

The installer adds the KernelPilot marketplace, installs `humanize@KernelPilot`,
links `KernelWiki` and `ncu-report-skill` into Claude Code's skills directory,
installs KernelWiki query dependencies, hydrates Claude Code's installed skill
cache with absolute `HUMANIZE_RUNTIME_ROOT`, `KERNELPILOT_ROOT`,
`KERNELWIKI_ROOT`, and `NCU_REPORT_SKILL_ROOT` paths, and fails if placeholders
remain.

Inside Claude Code, you should see commands such as
`/humanize:start-rlcr-loop` and skills such as `humanize-kernel-agent-loop`,
`KernelWiki`, and `ncu-report-skill`.

### Codex Install

```bash
humanize/scripts/install-skills-codex.sh
```

Generic installer:

```bash
humanize/scripts/install-skill.sh --target codex
```

After installation, restart the agent session and check that these skills are
available:

```text
humanize-kernel-agent-loop
KernelWiki
ncu-report-skill
```

## Prompt Card

```text
[$humanize-kernel-agent-loop] Optimize SGLang's int8_scaled_mm kernel on H100 for M=64, N=2048, K=2048, out_dtype=fp16, bias=true. Keep the work in a clean standalone repo, compare correctness and latency against the current SGLang baseline, and beat that baseline by at least 10% p50 latency on this focused case.
```

The stop hook summary should make the round outcome and review decision easy to
inspect:

![Humanize stop hook summary](docs/assets/humanize-stop-hook-summary.png)

The optimization ledger should make selected versions and rejected follow-ups
easy to scan:

![KernelPilot optimization ledger](docs/assets/kernelpilot-optimization-ledger.png)

## Maintenance

Validate external knowledge and installer wiring:

```bash
cd external/KernelWiki
python3 scripts/validate.py

cd ../..
humanize/tests/test-ncu-report-skill.sh
humanize/tests/run-all-tests.sh
```

Refresh or update submodules:

```bash
git submodule update --remote external/KernelWiki
git submodule update --remote external/ncu-report-skill
```

## Related

- [Humanize](https://github.com/PolyArch/humanize): the RLCR runtime that
  KernelPilot specializes for GPU kernel optimization.
- [KernelWiki](https://github.com/BBuf/KernelWiki/tree/kernelpilot-knowledge-expansion):
  the expanded GPU kernel evidence skill used by this repo.
- [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS):
  broader serving, profiling, SGLang, incident, and model optimization skills.
