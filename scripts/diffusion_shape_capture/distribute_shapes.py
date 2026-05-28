"""Copy captured shape JSONL files into each kernel task folder's docs/.

Maps each captured kernel.entry-name to the family folder that owns it. Drops
into both B200 and H200 task folders. Also writes a per-task `captured_shapes.md`
that summarizes the unique observed shape signatures.

Expects /tmp/shapes_h200_8.jsonl, /tmp/shapes_h200_9.jsonl, and (when ready)
/tmp/shapes_b200.jsonl to exist locally.

Usage:
    python3 distribute_shapes.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent  # set at exec-time to repo root
KERNELS_DIR = REPO_ROOT / "kernels"

CAPTURE_FILES = [
    Path("/tmp/shapes_h200_8.jsonl"),
    Path("/tmp/shapes_h200_9.jsonl"),
    Path("/tmp/shapes_b200.jsonl"),
]

# kernel-name -> task family slug
KERNEL_TO_FAMILY = {
    "qknorm_rope.fused_inplace_qknorm_rope": "diffusion_qknorm_rope__multi_shape",
    "norm.rms_norm_fn": "diffusion_rms_norm_fn__multi_shape",
    "norm.norm_infer": "diffusion_norm_infer__multi_shape",
    "rmsnorm_onepass.triton_one_pass_rms_norm": "diffusion_norm_infer__multi_shape",
    "group_norm_silu.triton_group_norm_silu": "diffusion_group_norm_silu__multi_shape",
    "group_norm_silu.apply_group_norm_silu": "diffusion_group_norm_silu__multi_shape",
    "rotary.apply_rotary_embedding": "diffusion_rotary_embedding__multi_shape",
    "ltx2_rotary.apply_ltx2_split_rotary_emb": "diffusion_rotary_embedding__multi_shape",
    "scale_shift.fuse_scale_shift_kernel": "diffusion_fuse_scale_shift__multi_shape",
    "scale_shift.fuse_layernorm_scale_shift_gate_select01_kernel": "diffusion_fuse_scale_shift__multi_shape",
    "scale_shift.fuse_residual_layernorm_scale_shift_gate_select01_kernel": "diffusion_fuse_scale_shift__multi_shape",
    "norm_tanh_mul_add_norm_scale.fused_norm_tanh_mul_add": "diffusion_cutedsl_norm_tanh_mul_add__multi_shape",
    "norm_tanh_mul_add_norm_scale.fused_norm_tanh_mul_add_norm_scale": "diffusion_cutedsl_norm_tanh_mul_add__multi_shape",
    "scale_residual_norm_scale_shift.fused_norm_scale_shift": "diffusion_cutedsl_norm_scale_shift__multi_shape",
    "scale_residual_norm_scale_shift.fused_scale_residual_norm_scale_shift": "diffusion_cutedsl_norm_scale_shift__multi_shape",
}


def load_records() -> list[dict]:
    out = []
    for p in CAPTURE_FILES:
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def shape_signature(rec: dict):
    args = rec.get("args") or []
    kwargs = rec.get("kwargs") or {}

    def norm(v):
        if isinstance(v, dict):
            if "shape" in v and "dtype" in v:
                return ("T", tuple(v["shape"]), v["dtype"])
            return tuple(sorted((k, norm(x)) for k, x in v.items()))
        if isinstance(v, list):
            return tuple(norm(x) for x in v)
        return v

    return (norm(args), tuple(sorted((k, norm(v)) for k, v in kwargs.items())))


def describe(value):
    if isinstance(value, dict) and "shape" in value:
        return f"`{value['shape']}/{value['dtype'].replace('torch.', '')}{'C' if value.get('contiguous') else 'NC'}`"
    if isinstance(value, list):
        return "[" + ", ".join(describe(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={describe(v)}" for k, v in value.items()) + "}"
    return f"`{value}`"


def main() -> None:
    records = load_records()
    print(f"Loaded {len(records)} capture records")

    # kernel -> arch -> family -> [unique signatures]
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    family_arch_records = defaultdict(lambda: defaultdict(list))

    for rec in records:
        if rec.get("event") == "install":
            continue
        kernel = rec.get("kernel")
        arch = rec.get("arch")
        if kernel is None or arch is None:
            continue
        family = KERNEL_TO_FAMILY.get(kernel)
        if family is None:
            continue
        sig = shape_signature(rec)
        if sig not in grouped[kernel][arch][family]:
            grouped[kernel][arch][family][sig] = rec
            family_arch_records[family][arch].append(rec)

    for family in {f for fam in KERNEL_TO_FAMILY.values() for f in [fam]}:
        for arch in ("b200", "h200"):
            task_slug = f"{arch}_{family}"
            folder = KERNELS_DIR / task_slug
            if not folder.exists():
                continue
            docs = folder / "docs"
            docs.mkdir(parents=True, exist_ok=True)

            # Copy raw captures for this family/arch
            raw_path = docs / f"captured_shapes_{arch}.jsonl"
            recs = family_arch_records[family].get(arch, [])
            # For B200 fallback to H200 captures if no B200 captures yet
            if not recs and arch == "b200":
                recs = family_arch_records[family].get("h200", [])
                annotation_arch = "h200 (shapes are arch-independent for this kernel)"
            else:
                annotation_arch = arch
            with raw_path.open("w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")

            # Write a markdown summary
            md_path = docs / f"captured_shapes_{arch}.md"
            with md_path.open("w") as md:
                md.write(f"# Captured shapes for `{task_slug}`\n\n")
                md.write(
                    f"Source: SGLang diffusion benchmark sweep captures from "
                    f"{annotation_arch}. Generated by `distribute_shapes.py`.\n\n"
                )
                if not recs:
                    md.write(
                        "No live captures were obtained for this kernel in the most "
                        "recent sweep. Fall back to the analytical shape table in "
                        "`prompt.md` and try `kernel_shape_capture.py` again with a "
                        "broader preset list.\n"
                    )
                    continue
                seen = set()
                md.write("| Model | Kernel | Tensor shapes (args[0..]) | Other args |\n")
                md.write("|---|---|---|---|\n")
                for r in recs:
                    sig = shape_signature(r)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    args = r.get("args") or []
                    kwargs = r.get("kwargs") or {}
                    tensor_parts = []
                    other_parts = []
                    for a in args:
                        if isinstance(a, dict) and "shape" in a:
                            tensor_parts.append(describe(a))
                        else:
                            other_parts.append(describe(a))
                    # kwargs - separate tensors from scalars
                    kw_tensor_parts = []
                    kw_other_parts = []
                    for k, v in kwargs.items():
                        if isinstance(v, dict) and "shape" in v:
                            kw_tensor_parts.append(f"{k}={describe(v)}")
                        else:
                            kw_other_parts.append(f"{k}={describe(v)}")
                    tensor_md = " ; ".join(tensor_parts + kw_tensor_parts) or "(none)"
                    other_md = " ; ".join(other_parts + kw_other_parts) or "(none)"
                    md.write(
                        f"| {r.get('model')} | `{r.get('kernel')}` | {tensor_md} | {other_md} |\n"
                    )
                md.write(
                    "\nLegend: `<shape>/<dtype>/C|NC` where `C`=contiguous, `NC`=non-contiguous.\n"
                )

    # Also write the top-level shape ledger
    top = KERNELS_DIR.parent / "kernels" / "diffusion_shapes_ledger.md"
    with top.open("w") as f:
        f.write("# Diffusion Shape Ledger (captured from SGLang benchmark presets)\n\n")
        f.write(
            "This file is the cross-task summary of shapes captured from the SGLang\n"
            "diffusion benchmark sweep, aggregated across `ion-b200`, `ion8-h200`, and\n"
            "`ion9-h200`. The per-task `docs/captured_shapes_<arch>.md` files carry the\n"
            "same data scoped to one kernel family.\n\n"
        )
        for kernel in sorted(grouped):
            f.write(f"## `{kernel}`\n\n")
            f.write("| Arch | Model | Tensor shapes | Other args |\n")
            f.write("|---|---|---|---|\n")
            for arch in ("b200", "h200"):
                for family, recs in sorted(grouped[kernel][arch].items()):
                    for rec in recs.values():
                        args = rec.get("args") or []
                        kwargs = rec.get("kwargs") or {}
                        tensor_md = " ; ".join(
                            describe(a) for a in args
                            if isinstance(a, dict) and "shape" in a
                        ) + " ; " + " ; ".join(
                            f"{k}={describe(v)}" for k, v in kwargs.items()
                            if isinstance(v, dict) and "shape" in v
                        )
                        other_md = " ; ".join(
                            describe(a) for a in args
                            if not isinstance(a, dict) or "shape" not in a
                        ) + " ; " + " ; ".join(
                            f"{k}={describe(v)}" for k, v in kwargs.items()
                            if not isinstance(v, dict) or "shape" not in v
                        )
                        f.write(
                            f"| {arch} | {rec.get('model')} | {tensor_md.strip(' ;')} | {other_md.strip(' ;')} |\n"
                        )
            f.write("\n")
    print(f"Wrote {top}")


if __name__ == "__main__":
    main()
