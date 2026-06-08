#!/usr/bin/env python3
"""Extract a kernel-workflow inventory from a torch/Kineto profiler trace.

Aggregates GPU kernel time by name, ranks by share of total GPU kernel time,
categorizes each kernel, and emits a Markdown table + CSV. Attention and cuDNN
kernels are reported only as an aggregate (never as optimization tasks); every
other kernel at or above the threshold (default 1%) is an acceleration
opportunity.

Usage:
  extract_kernel_workflow.py <trace.json[.gz] | trace_dir> \
      --out-md docs/kernel_workflow.md --out-csv docs/kernel_workflow.csv \
      [--threshold 1.0] [--label "MiniMax-M2.7 / B200 / mid-conc"]
"""
import argparse
import csv
import glob
import gzip
import io
import json
import os
import re
import sys
from collections import defaultdict

# ---- categorization (order matters: first match wins) ----------------------
# (category, is_excluded, list of lowercased substring/regex patterns)
RULES = [
    ("attention", True,  ["flash", "fmha", "fa2", "fa3", "fa4", "mla", "mha",
                           "attention", "attn", "paged", "lightning_indexer",
                           "sparse_attn", "decode_attention", "extend_attention"]),
    ("cudnn",     True,  ["cudnn"]),
    ("comm",      False, ["nccl", "allreduce", "all_reduce", "all_gather",
                           "allgather", "reduce_scatter", "reducescatter",
                           "alltoall", "all_to_all", "a2a", "sendrecv"]),
    ("moe",       False, ["moe", "expert", "grouped_gemm", "group_gemm",
                           "groupgemm", "fused_moe", "topk", "moe_align",
                           "moe_sum", "cutlass_grouped", "grouped_mm"]),
    ("quant_gemm",False, ["fp8", "int8", "nvfp4", "mxfp4", "fp4", "scaled_mm",
                           "w8a8", "w4a16", "w4a8", "marlin", "machete", "awq",
                           "gptq", "blockwise", "per_token_quant",
                           "per_tensor_quant", "quant_gemm"]),
    ("gemm",      False, ["cutlass", "cublas", "gemm", "sgemm", "hgemm",
                           "matmul", "cijk", "wgrad", "ampere_", "sm80_", "sm90_",
                           "sm100_", "tensorop", "gemv"]),
    ("rope",      False, ["rope", "rotary"]),
    ("norm",      False, ["rmsnorm", "rms_norm", "layernorm", "layer_norm",
                           "norm_kernel", "groupnorm"]),
    ("memory_bound", False, ["elementwise", "vectorized_elementwise", "silu",
                           "gelu", "activation", "act_and_mul", "residual",
                           "add_kernel", "copy", "cast", "dequant", "quantize",
                           "reduce", "embedding", "scatter", "gather",
                           "index", "fill", "memset", "transpose", "concat",
                           "rotary_embedding", "sampling", "softmax"]),
]


def classify(name: str):
    low = name.lower()
    for cat, excluded, pats in RULES:
        for p in pats:
            if p in low:
                return cat, excluded
    return "other", False


def load_trace(path: str):
    """Return list of trace events from a .json / .json.gz file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    obj = json.loads(data.decode("utf-8"))
    if isinstance(obj, dict):
        return obj.get("traceEvents", [])
    return obj  # some traces are a bare list


def resolve_trace(arg: str) -> str:
    if os.path.isdir(arg):
        cands = []
        for pat in ("*.pt.trace.json.gz", "*.trace.json.gz", "*.trace.json",
                    "*.json.gz", "*.json"):
            cands += glob.glob(os.path.join(arg, "**", pat), recursive=True)
        cands = sorted(set(cands), key=os.path.getmtime)
        if not cands:
            sys.exit(f"no trace files found under {arg}")
        return cands[-1]
    return arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="trace .json/.json.gz file or a directory")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="min %% of total GPU kernel time to record (default 1.0)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    trace_path = resolve_trace(args.trace)
    events = load_trace(trace_path)

    # GPU kernels in Kineto traces carry cat == "kernel".
    dur = defaultdict(float)   # name -> total us
    cnt = defaultdict(int)
    for e in events:
        if not isinstance(e, dict):
            continue
        cat = str(e.get("cat", "")).lower()
        if cat != "kernel":
            continue
        name = e.get("name", "")
        d = e.get("dur", 0) or 0
        dur[name] += float(d)
        cnt[name] += 1

    if not dur:
        sys.exit(f"no GPU kernel events (cat='kernel') in {trace_path}")

    total = sum(dur.values())
    rows = []
    for name, us in dur.items():
        cat, excluded = classify(name)
        rows.append({
            "name": name,
            "category": cat,
            "excluded": excluded,
            "calls": cnt[name],
            "total_us": us,
            "pct": 100.0 * us / total,
            "mean_us": us / max(cnt[name], 1),
        })
    rows.sort(key=lambda r: r["total_us"], reverse=True)

    kept = [r for r in rows if (not r["excluded"]) and r["pct"] >= args.threshold]
    excluded_total = sum(r["pct"] for r in rows if r["excluded"])
    attn_total = sum(r["pct"] for r in rows if r["category"] == "attention")
    cudnn_total = sum(r["pct"] for r in rows if r["category"] == "cudnn")

    # ---- CSV (all kernels, with kept/excluded flag) ----
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "name", "category", "excluded", "calls",
                    "total_us", "pct_of_gpu", "mean_us", "is_opportunity"])
        for i, r in enumerate(rows, 1):
            is_opp = (not r["excluded"]) and r["pct"] >= args.threshold
            w.writerow([i, r["name"], r["category"], r["excluded"], r["calls"],
                        f"{r['total_us']:.1f}", f"{r['pct']:.3f}",
                        f"{r['mean_us']:.2f}", is_opp])

    # ---- Markdown inventory (kept opportunities) ----
    by_cat = defaultdict(float)
    for r in kept:
        by_cat[r["category"]] += r["pct"]

    def short(n, k=70):
        return n if len(n) <= k else n[:k - 1] + "…"

    lines = []
    lines.append(f"# Kernel-Workflow Inventory{(' — ' + args.label) if args.label else ''}")
    lines.append("")
    lines.append(f"- Trace: `{os.path.basename(trace_path)}`")
    lines.append(f"- Total GPU kernel time: {total/1000:.1f} ms across {len(rows)} distinct kernels")
    lines.append(f"- Threshold: ≥ {args.threshold:.1f}% of GPU kernel time")
    lines.append(f"- **Excluded (not tasks):** attention {attn_total:.1f}% + cuDNN {cudnn_total:.1f}% = {excluded_total:.1f}% of GPU time")
    lines.append(f"- Opportunities recorded: {len(kept)} kernels = {sum(r['pct'] for r in kept):.1f}% of GPU time")
    lines.append("")
    lines.append("## Opportunity share by category")
    lines.append("")
    lines.append("| Category | % of GPU time |")
    lines.append("|---|---|")
    for cat, pct in sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"| {cat} | {pct:.1f}% |")
    lines.append("")
    lines.append("## Opportunity kernels (≥ threshold, attention/cuDNN excluded)")
    lines.append("")
    lines.append("| # | % | calls | mean µs | category | kernel |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(kept, 1):
        lines.append(f"| {i} | {r['pct']:.2f} | {r['calls']} | {r['mean_us']:.1f} | "
                     f"{r['category']} | `{short(r['name'])}` |")
    lines.append("")
    lines.append("> Full per-kernel data (including excluded attention/cuDNN) is in the sibling CSV.")
    lines.append("")

    with open(args.out_md, "w") as f:
        f.write("\n".join(lines))

    print(f"trace:        {trace_path}")
    print(f"total GPU:    {total/1000:.1f} ms, {len(rows)} kernels")
    print(f"excluded:     attention {attn_total:.1f}% + cudnn {cudnn_total:.1f}%")
    print(f"opportunities: {len(kept)} kernels >= {args.threshold}%")
    print(f"wrote:        {args.out_md}")
    print(f"wrote:        {args.out_csv}")


if __name__ == "__main__":
    main()
