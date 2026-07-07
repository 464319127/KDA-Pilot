#!/usr/bin/env python3
"""Aggregate KDA SGLang kernel shape-capture JSONL into task workload files."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


TASK_RULES = {
    "glm_52__sglang_deep_gemm_fp8_fp8_bf16_nt": (
        "deep_gemm_fp8_fp8_bf16_nt",
    ),
    "glm_52__fp8_bmm": (
        ".bmm_fp8",
        "._bmm_fp8_op",
        ".flashinfer_bmm_fp8",
        "torch.bmm",
        "fp8_paged_mqa_logits_torch",
        "_aiter_fp8_paged_mqa_logits",
    ),
    "glm_52__sglang_unified_attention_with_output": (
        "unified_attention_with_output",
        "mla_bmm_then_unified_attention",
        "DeepseekV4AttnBackend.forward",
        "DeepseekV4HipRadixBackend.forward",
        "DeepseekSparseAttnBackend.forward_extend",
        "DeepseekSparseAttnBackend.forward_decode",
        "DeepseekSparseAttnBackend._forward_fa3",
        "DeepseekSparseAttnBackend._forward_flashmla_sparse",
        "DeepseekSparseAttnBackend._forward_flashmla_kv",
        "DeepseekSparseAttnBackend._forward_standard_mha",
        "DeepseekSparseAttnBackend._forward_trtllm",
        "flash_mla_with_kvcache",
        "flash_mla_sparse_fwd",
        "trtllm_ragged_attention_deepseek",
        "trtllm_batch_decode_with_kv_cache_mla",
        "_forward_prefill_sparse",
    ),
    "glm_52__per_token_group_quant": (
        "sglang_per_token_group_quant_fp8",
        "per_token_group_quant_fp8",
        "sglang_per_token_group_quant_8bit",
        "per_token_group_quant_8bit",
    ),
}

DROP_TENSOR_KEYS = {"device", "device_index", "device_type", "requires_grad", "numel"}


def load_jsonl(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: {exc}") from exc
                    row["_source_file"] = path
                    row["_source_line"] = line_no
                    rows.append(row)
    rows.sort(key=lambda row: (row.get("time", 0), row.get("pid", 0), row.get("call_index", 0)))
    return rows


def load_markers(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    markers: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                markers.append(json.loads(line))
    return markers


def scenario_for_time(markers: list[dict[str, Any]], timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    for marker in markers:
        start = marker.get("start_time")
        end = marker.get("end_time")
        if start is not None and end is not None and start <= timestamp <= end:
            return marker.get("scenario")
    return None


def tensor_paths(value: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict) and value.get("kind") == "tensor":
        out.append((prefix, value))
    elif isinstance(value, dict) and value.get("kind") in {"list", "tuple"}:
        for idx, item in enumerate(value.get("items", [])):
            out.extend(tensor_paths(item, f"{prefix}.{idx}" if prefix else str(idx)))
    elif isinstance(value, dict) and value.get("kind") == "dict":
        for key, item in value.get("items", {}).items():
            out.extend(tensor_paths(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            out.extend(tensor_paths(item, f"{prefix}.{idx}" if prefix else str(idx)))
    return out


def scalarize(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("kind") == "tensor":
            return {
                key: val
                for key, val in value.items()
                if key not in DROP_TENSOR_KEYS
            }
        if value.get("kind") in {"list", "tuple"}:
            return [scalarize(item) for item in value.get("items", [])]
        if value.get("kind") == "dict":
            return {key: scalarize(item) for key, item in value.get("items", {}).items()}
        if "repr" in value:
            return {"kind": value.get("kind")}
        return value
    if isinstance(value, list):
        return [scalarize(item) for item in value]
    return value


def record_signature(row: dict[str, Any]) -> str:
    material = {
        "function": row.get("function"),
        "args": scalarize(row.get("args")),
        "kwargs": scalarize(row.get("kwargs")),
    }
    return json.dumps(material, sort_keys=True, separators=(",", ":"))


def task_for_function(function: str) -> str | None:
    for task, needles in TASK_RULES.items():
        if any(needle in function for needle in needles):
            return task
    return None


def compact_workload(row: dict[str, Any], scenario: str | None, ordinal: int) -> dict[str, Any]:
    function = row["function"]
    arg_tensors = OrderedDict(tensor_paths(row.get("args", []), "arg"))
    kwarg_tensors = OrderedDict(tensor_paths(row.get("kwargs", {}), "kwarg"))
    tensors = OrderedDict()
    tensors.update((key, scalarize(meta)) for key, meta in arg_tensors.items())
    tensors.update((key, scalarize(meta)) for key, meta in kwarg_tensors.items())
    shape_tag = "_".join(
        "x".join(str(dim) for dim in meta.get("shape", []))
        for meta in list(tensors.values())[:3]
        if meta.get("shape") is not None
    )
    shape_tag = re.sub(r"[^0-9x]+", "_", shape_tag).strip("_")[:80] or "scalar"
    return {
        "id": f"{function.rsplit('.', 1)[-1]}__{scenario or 'unmarked'}__{ordinal:04d}__{shape_tag}",
        "production": True,
        "source": "fresh_sglang_kernel_api_capture",
        "scenario": scenario,
        "function": function,
        "tensors": tensors,
        "args": scalarize(row.get("args", [])),
        "kwargs": scalarize(row.get("kwargs", {})),
        "result": scalarize(row.get("result")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--markers", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model", default="zai-org/GLM-5.2-FP8")
    parser.add_argument("--capture-note", default="")
    args = parser.parse_args()

    rows = load_jsonl(args.records)
    markers = load_markers(args.markers)
    by_task: dict[str, OrderedDict[str, dict[str, Any]]] = {
        task: OrderedDict() for task in TASK_RULES
    }
    unmatched = 0

    for row in rows:
        function = row.get("function")
        if not function:
            unmatched += 1
            continue
        task = task_for_function(function)
        if task is None:
            unmatched += 1
            continue
        scenario = scenario_for_time(markers, row.get("time"))
        signature = record_signature(row)
        if signature not in by_task[task]:
            by_task[task][signature] = compact_workload(row, scenario, len(by_task[task]) + 1)

    summary: dict[str, Any] = {
        "model": args.model,
        "record_count": len(rows),
        "unmatched_record_count": unmatched,
        "marker_count": len(markers),
        "capture_note": args.capture_note,
        "tasks": {},
    }

    for task, workloads_by_sig in by_task.items():
        workloads = list(workloads_by_sig.values())
        task_dir = args.repo_root / "llm" / task
        docs_dir = task_dir / "docs"
        bench_dir = task_dir / "bench"
        docs_dir.mkdir(parents=True, exist_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "task": task,
            "source": "fresh_sglang_kernel_api_capture",
            "capture_note": args.capture_note,
            "markers": markers,
            "workload_count": len(workloads),
            "workloads": workloads,
        }
        (docs_dir / "captured_kernel_api_shapes.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        (bench_dir / "workloads.json").write_text(
            json.dumps(workloads, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["tasks"][task] = {
            "workload_count": len(workloads),
            "functions": sorted({w["function"] for w in workloads}),
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
