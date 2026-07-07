#!/usr/bin/env python3
"""Refresh task metadata after kernel API shape capture aggregation."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path
from typing import Any


CONCURRENCY = {"low": "1", "mid": "32", "high": "100"}
SCENARIO_ORDER = (
    "random_low",
    "random_mid",
    "random_high",
    "sharegpt_low",
    "sharegpt_mid",
    "sharegpt_high",
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def task_dirs(repo_root: Path, task_prefix: str) -> list[Path]:
    return sorted((repo_root / "llm").glob(f"{task_prefix}__*"))


def workload_functions(task_dir: Path) -> list[str]:
    workloads = load_json(task_dir / "bench" / "workloads.json")
    return sorted({row["function"] for row in workloads})


def update_config(task_dir: Path, functions: list[str]) -> None:
    path = task_dir / "config.toml"
    text = path.read_text(encoding="utf-8")
    entry_block = "entry_points = [\n" + "".join(
        f'  "{function}",\n' for function in functions
    ) + "]"
    text = re.sub(r"entry_points = \[\n(?:  .*\n)*?\]", entry_block, text, count=1)
    if "shape_source_json" not in text:
        text = text.replace(
            'evidence_json = "docs/profile_evidence.json"\n',
            'evidence_json = "docs/profile_evidence.json"\n'
            'shape_source_json = "docs/captured_kernel_api_shapes.json"\n'
            'workloads_json = "bench/workloads.json"\n',
            1,
        )
    else:
        text = re.sub(
            r'shape_source_json = ".*"',
            'shape_source_json = "docs/captured_kernel_api_shapes.json"',
            text,
            count=1,
        )
        text = re.sub(
            r'workloads_json = ".*"',
            'workloads_json = "bench/workloads.json"',
            text,
            count=1,
        )
    path.write_text(text, encoding="utf-8")


def scenario_rows(per: dict[str, float]) -> str:
    rows: list[str] = []
    labels = list(SCENARIO_ORDER) + sorted(set(per) - set(SCENARIO_ORDER))
    for label in labels:
        if label not in per:
            continue
        dataset, level = label.rsplit("_", 1)
        rows.append(f"| {dataset} | conc {CONCURRENCY.get(level, level)} | {per[label]:.2f}% |")
    return "\n".join(rows)


def refresh_profile_evidence(task_dir: Path, capture_note: str) -> dict[str, Any]:
    evidence_path = task_dir / "docs" / "profile_evidence.json"
    evidence = load_json(evidence_path)
    captured = load_json(task_dir / "docs" / "captured_kernel_api_shapes.json")
    functions = workload_functions(task_dir)
    evidence["python_interface"] = functions
    evidence["input_shapes"] = []
    evidence["input_shapes_replaced_by"] = "docs/captured_kernel_api_shapes.json"
    evidence["standalone_workloads_json"] = "bench/workloads.json"
    evidence["shape_source"] = "fresh_sglang_kernel_api_capture"
    evidence["captured_kernel_api_workload_count"] = captured["workload_count"]
    evidence["captured_kernel_api_functions"] = functions
    evidence["shape_capture_note"] = capture_note
    dump_json(evidence_path, evidence)
    return evidence


def write_profile_md(task_dir: Path, evidence: dict[str, Any]) -> None:
    task = task_dir.name
    functions = evidence["captured_kernel_api_functions"]
    per = evidence.get("pct_of_gpu_by_scenario", {})
    kernels = evidence.get("gpu_kernels", [])
    note = evidence.get("shape_capture_note", "")
    md = f"""# Profile evidence - {task}

**Standalone kernel target: {evidence.get("max_pct_of_gpu", 0):.1f}% of total serving GPU time** (max across scenarios) on
`{evidence.get("model")}`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below were recaptured from a real `{evidence.get("model")}` server run and replace the old noisy profiler shape strings.

- Model: `{evidence.get("model")}` (slug `{evidence.get("model_slug")}`, tp={evidence.get("tp")})
- Python interface(s): {", ".join(f"`{function}`" for function in functions)}
- Kernel family: `{evidence.get("kernel_family")}`  .  Category: `{evidence.get("category")}`
- GPU kernel(s): {", ".join(f"`{kernel}`" for kernel in kernels)}

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
{scenario_rows(per)}

**Peak: {evidence.get("max_pct_of_gpu", 0):.1f}% in `{evidence.get("best_scenario")}`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: {evidence.get("captured_kernel_api_workload_count")}
- Capture note: {note}

Functions covered:
{chr(10).join(f"- `{function}`" for function in functions)}

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Original serving profile command (provenance only)
```bash
{evidence.get("cookbook_cmd")}
```
This command is retained only to explain target selection. Normal RLCR kernel
work must not depend on a live SGLang server, `run_capture`, 8-GPU availability,
or a multi-GPU e2e gate. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set. Re-run serving capture only
when intentionally refreshing these evidence files.
"""
    (task_dir / "docs" / "profile_evidence.md").write_text(md, encoding="utf-8")


def write_prompt(task_dir: Path, evidence: dict[str, Any]) -> None:
    with (task_dir / "config.toml").open("rb") as f:
        config = tomllib.load(f)
    task_cfg = config["task"]
    functions = evidence["captured_kernel_api_functions"]
    prompt = f"""# KDA Prompt: {task_dir.name}

Target GPU: {task_cfg.get("target_gpu", "NVIDIA B200")}. Optimize the SGLang kernel path behind:

{chr(10).join(f"- `{function}`" for function in functions)}

**{evidence.get("max_pct_of_gpu", 0):.1f}% of total serving GPU time** on `{evidence.get("model")}` (cookbook-aligned
profile, peak `{evidence.get("best_scenario")}`) - a serving-profile headroom signal used to select this
standalone kernel task. Family `{evidence.get("kernel_family")}`, category `{evidence.get("category")}`.

Use `bench/workloads.json` as the task-local standalone shape source. It was generated from
`docs/captured_kernel_api_shapes.json`, a fresh real `{evidence.get("model")}` TP={evidence.get("tp")} SGLang capture. Normal
RLCR kernel work must not depend on starting SGLang serve, `run_capture`, 8-GPU availability,
or a multi-GPU e2e A/B; optimize and validate via the task-local standalone benchmark on
one idle target GPU. Follow
`llm/docs/llm_kernel_optimization_rules.md` (CUDA, no DSL) + `llm/docs/llm_correctness_contract.md`.
"""
    (task_dir / "prompt.md").write_text(prompt, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task-prefix", required=True)
    parser.add_argument("--capture-note", required=True)
    args = parser.parse_args()

    refreshed: dict[str, Any] = {}
    for task_dir in task_dirs(args.repo_root, args.task_prefix):
        capture_path = task_dir / "docs" / "captured_kernel_api_shapes.json"
        workloads_path = task_dir / "bench" / "workloads.json"
        if not capture_path.exists() or not workloads_path.exists():
            continue
        functions = workload_functions(task_dir)
        if not functions:
            raise SystemExit(f"{task_dir.name}: no captured functions")
        update_config(task_dir, functions)
        evidence = refresh_profile_evidence(task_dir, args.capture_note)
        write_profile_md(task_dir, evidence)
        write_prompt(task_dir, evidence)
        refreshed[task_dir.name] = {
            "workload_count": evidence["captured_kernel_api_workload_count"],
            "functions": functions,
        }

    if not refreshed:
        raise SystemExit(f"no captured task artifacts found for {args.task_prefix}")
    print(json.dumps(refreshed, indent=2))


if __name__ == "__main__":
    main()
