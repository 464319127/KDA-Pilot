"""Aggregate captured shape JSONL into per-kernel-per-arch summary tables.

Reads /tmp/shapes_*.jsonl (h200_8, h200_9, b200) and emits:
  - /tmp/shapes_summary.json  : kernel -> arch -> [unique-shape-records]
  - /tmp/shapes_summary.md     : human-readable per-kernel table
  - Per-kernel copy files at /tmp/shapes_<kernel>_<arch>.jsonl
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

INPUTS = [
    ("h200", Path("/tmp/shapes_h200_8.jsonl")),
    ("h200", Path("/tmp/shapes_h200_9.jsonl")),
    ("b200", Path("/tmp/shapes_b200.jsonl")),
]


def shape_signature(rec: dict) -> tuple:
    """Build a stable signature for deduplication."""
    args = rec.get("args") or []
    kwargs = rec.get("kwargs") or {}
    def norm(v):
        if isinstance(v, dict):
            if "shape" in v and "dtype" in v:
                return ("T", tuple(v["shape"]), v["dtype"], v.get("contiguous"))
            return tuple(sorted((k, norm(x)) for k, x in v.items()))
        if isinstance(v, list):
            return tuple(norm(x) for x in v)
        return v
    return (norm(args), tuple(sorted((k, norm(v)) for k, v in kwargs.items())))


def main() -> None:
    by_kernel_arch_sig = defaultdict(lambda: defaultdict(dict))
    install_events = defaultdict(set)
    for arch, path in INPUTS:
        if not path.exists():
            continue
        with path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except Exception:
                    continue
                if rec.get("event") == "install":
                    install_events[(arch, rec.get("model"))].add(rec.get("host"))
                    continue
                kernel = rec.get("kernel")
                if not kernel:
                    continue
                sig = shape_signature(rec)
                key = by_kernel_arch_sig[kernel][arch]
                if sig not in key:
                    key[sig] = {
                        "kernel": kernel,
                        "arch": arch,
                        "model": rec.get("model"),
                        "host": rec.get("host"),
                        "first_call_idx": rec.get("call_idx"),
                        "args": rec.get("args"),
                        "kwargs": rec.get("kwargs"),
                    }

    summary = {}
    for kernel, by_arch in by_kernel_arch_sig.items():
        summary[kernel] = {}
        for arch, sigs in by_arch.items():
            summary[kernel][arch] = list(sigs.values())

    Path("/tmp/shapes_summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Captured shape summary\n"]
    lines.append("Sweep coverage by (arch, model):\n")
    for (arch, model), hosts in sorted(install_events.items()):
        lines.append(f"- {arch} / {model}: hosts {sorted(hosts)}")
    lines.append("\n")
    for kernel in sorted(summary):
        lines.append(f"## {kernel}\n")
        for arch in ("b200", "h200"):
            recs = summary[kernel].get(arch, [])
            if not recs:
                lines.append(f"- **{arch}**: no captures yet\n")
                continue
            lines.append(f"- **{arch}** ({len(recs)} unique shape signatures):")
            seen_models = sorted({r["model"] for r in recs})
            lines.append(f"  - models: {seen_models}")
            shape_strings = set()
            for r in recs:
                args = r.get("args") or []
                tensor_shapes = []
                for a in args:
                    if isinstance(a, dict) and "shape" in a:
                        tensor_shapes.append(f"{a['shape']}/{a['dtype'].replace('torch.', '')}")
                key = " | ".join(tensor_shapes[:3])
                shape_strings.add(key)
            for s in sorted(shape_strings)[:40]:
                lines.append(f"    - {s}")
        lines.append("")

    Path("/tmp/shapes_summary.md").write_text("\n".join(lines))
    print("Wrote /tmp/shapes_summary.json and /tmp/shapes_summary.md")
    print(f"Total kernel categories: {len(summary)}")


if __name__ == "__main__":
    main()
