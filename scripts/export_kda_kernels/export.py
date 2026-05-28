#!/usr/bin/env python3
"""Promote a KDA kernel task into kda_kernels/.

Usage:
  scripts/export_kda_kernels/export.py <task-slug>
  scripts/export_kda_kernels/export.py --list
  scripts/export_kda_kernels/export.py --revert <task-slug>

`<task-slug>` is the folder name under `kernels/`, for example
`b200_diffusion_qknorm_rope__multi_shape`. Both b200 and h200 slugs of the
same family are accepted; the family they share is what gets wired into
kda_kernels, so promoting either is enough (and promoting both is a no-op
the second time).

What the export does:

1. Copies `kernels/<task-slug>/src/*` into
   `kda_kernels/_impls/<task-slug>/`. CUDA sources (`.cu`, `.cuh`, `.cpp`,
   `.h`), Python wrappers, and any extra helpers ride along.
2. Reads the task's `src/register.py` for its `EXPORTS` dict. Each
   `(function_name, callable)` pair is a promised swap target.
3. Rewrites the matching kda_kernels stub file at the kernel-family
   target (e.g. `kda_kernels/diffusion/qknorm_rope.py`) so that each
   promoted function imports from the new impl subpackage and the
   corresponding `KDA_OPTIMIZED_<fn>` flag flips to True.
4. Stamps `KDA_TASK_<fn>`, `KDA_COMMIT_<fn>`, `KDA_DATE_<fn>`, and (when
   present in the task's `benchmark.csv` last row) `KDA_SPEEDUP_<fn>`.

Re-running export.py for the same task overwrites the impl copy with the
current task src/. Use `--revert <task-slug>` to undo (the stub falls back
to re-exporting the sglang baseline with `KDA_OPTIMIZED_<fn> = False`).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib.util
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KDA = REPO / "kda_kernels"
IMPLS = KDA / "_impls"
KERNELS = REPO / "kernels"

# Owning task family -> list of (sglang_module, sglang_function,
# kda_module, kda_function).
FAMILY_TO_SWAPS: dict[str, list[tuple[str, str, str, str]]] = {
    "qknorm_rope": [
        ("sglang.jit_kernel.diffusion.qknorm_rope", "fused_inplace_qknorm_rope",
         "kda_kernels.diffusion.qknorm_rope", "fused_inplace_qknorm_rope"),
    ],
    "rms_norm_fn": [
        ("sglang.jit_kernel.diffusion.triton.norm", "rms_norm_fn",
         "kda_kernels.diffusion.triton.norm", "rms_norm_fn"),
    ],
    "norm_infer": [
        ("sglang.jit_kernel.diffusion.triton.norm", "norm_infer",
         "kda_kernels.diffusion.triton.norm", "norm_infer"),
        ("sglang.jit_kernel.diffusion.triton.rmsnorm_onepass", "triton_one_pass_rms_norm",
         "kda_kernels.diffusion.triton.rmsnorm_onepass", "triton_one_pass_rms_norm"),
    ],
    "group_norm_silu": [
        ("sglang.jit_kernel.diffusion.triton.group_norm_silu", "triton_group_norm_silu",
         "kda_kernels.diffusion.triton.group_norm_silu", "triton_group_norm_silu"),
        ("sglang.jit_kernel.diffusion.group_norm_silu", "apply_group_norm_silu",
         "kda_kernels.diffusion.group_norm_silu", "apply_group_norm_silu"),
    ],
    "rotary_embedding": [
        ("sglang.jit_kernel.diffusion.triton.rotary", "apply_rotary_embedding",
         "kda_kernels.diffusion.triton.rotary", "apply_rotary_embedding"),
        ("sglang.jit_kernel.diffusion.triton.ltx2_rotary", "apply_ltx2_split_rotary_emb",
         "kda_kernels.diffusion.triton.ltx2_rotary", "apply_ltx2_split_rotary_emb"),
    ],
    "fuse_scale_shift": [
        ("sglang.jit_kernel.diffusion.triton.scale_shift", "fuse_scale_shift_kernel",
         "kda_kernels.diffusion.triton.scale_shift", "fuse_scale_shift_kernel"),
        ("sglang.jit_kernel.diffusion.triton.scale_shift", "fuse_layernorm_scale_shift_gate_select01_kernel",
         "kda_kernels.diffusion.triton.scale_shift", "fuse_layernorm_scale_shift_gate_select01_kernel"),
        ("sglang.jit_kernel.diffusion.triton.scale_shift", "fuse_residual_layernorm_scale_shift_gate_select01_kernel",
         "kda_kernels.diffusion.triton.scale_shift", "fuse_residual_layernorm_scale_shift_gate_select01_kernel"),
    ],
    "cutedsl_norm_tanh_mul_add": [
        ("sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale", "fused_norm_tanh_mul_add",
         "kda_kernels.diffusion.cutedsl.norm_tanh_mul_add_norm_scale", "fused_norm_tanh_mul_add"),
        ("sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale", "fused_norm_tanh_mul_add_norm_scale",
         "kda_kernels.diffusion.cutedsl.norm_tanh_mul_add_norm_scale", "fused_norm_tanh_mul_add_norm_scale"),
    ],
    "cutedsl_norm_scale_shift": [
        ("sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift", "fused_norm_scale_shift",
         "kda_kernels.diffusion.cutedsl.scale_residual_norm_scale_shift", "fused_norm_scale_shift"),
        ("sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift", "fused_scale_residual_norm_scale_shift",
         "kda_kernels.diffusion.cutedsl.scale_residual_norm_scale_shift", "fused_scale_residual_norm_scale_shift"),
    ],
}


def family_of(task_slug: str) -> str:
    if "_diffusion_" not in task_slug or "__multi_shape" not in task_slug:
        raise SystemExit(
            f"task slug {task_slug!r} is not a diffusion multi-shape task"
        )
    family = task_slug.split("_diffusion_", 1)[1].split("__multi_shape", 1)[0]
    if family not in FAMILY_TO_SWAPS:
        raise SystemExit(f"unknown family {family!r} (slug={task_slug!r})")
    return family


def last_speedup(task_dir: Path) -> str:
    csv_path = task_dir / "benchmark.csv"
    if not csv_path.exists():
        return ""
    try:
        with csv_path.open() as f:
            rows = list(csv.reader(f))
        for row in reversed(rows):
            for cell in row:
                if cell.endswith("x") and any(c.isdigit() for c in cell):
                    return cell
    except Exception:  # noqa: BLE001
        pass
    return ""


def read_exports(src_dir: Path) -> set[str]:
    """Return the set of function names listed in src/register.py's EXPORTS dict."""
    register_py = src_dir / "register.py"
    if not register_py.exists():
        return set()
    ns: dict[str, object] = {}
    try:
        code = register_py.read_text()
        exec(compile(code, str(register_py), "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        print(
            f"warning: failed to read EXPORTS from {register_py}: {exc!r}",
            file=sys.stderr,
        )
        return set()
    exports = ns.get("EXPORTS")
    if isinstance(exports, dict):
        return {str(k) for k in exports}
    return set()


def copy_src(task_dir: Path, task_slug: str) -> Path:
    dst = IMPLS / task_slug
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(task_dir / "src", dst)
    init = dst / "__init__.py"
    if not init.exists():
        init.write_text(f'"""Promoted source for KDA task {task_slug}."""\n')
    return dst


def regenerate_stub(family: str, task_slug: str, exports: set[str]) -> list[Path]:
    """Write the kda_kernels stubs for every entry of `family`.

    Functions present in `exports` get a `from kda_kernels._impls.<task>.register
    import <fn>` line and `KDA_OPTIMIZED_<fn> = True`. Others fall back to
    re-exporting the sglang baseline with `KDA_OPTIMIZED_<fn> = False`.
    """
    written: list[Path] = []
    per_kda_mod: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for sgl_mod, _sgl_fn, kda_mod, kda_fn in FAMILY_TO_SWAPS[family]:
        per_kda_mod[kda_mod].append((sgl_mod, kda_fn, kda_fn))

    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except subprocess.CalledProcessError:
        pass
    date = _dt.date.today().isoformat()
    speedup = last_speedup(KERNELS / task_slug)

    for kda_mod, entries in per_kda_mod.items():
        rel = kda_mod.replace(".", "/") + ".py"
        path = REPO / rel
        sgl_mod = entries[0][0]
        lines: list[str] = [
            f'"""kda_kernels mirror of `{sgl_mod}`.\n\n',
            'Stub status: each function below is either re-exported from sglang\n',
            '(KDA_OPTIMIZED_<fn>=False) or pulled from a promoted KDA impl\n',
            '(KDA_OPTIMIZED_<fn>=True). See scripts/export_kda_kernels/export.py\n',
            'for the swap rule.\n"""\n\n',
        ]
        imports: list[str] = []
        flags: list[str] = []
        stamps: list[str] = []
        for _sgl, _fn, kda_fn in entries:
            if kda_fn in exports:
                imports.append(
                    f"from kda_kernels._impls.{task_slug}.register import "
                    f"{kda_fn}  # noqa: F401\n"
                )
                flags.append(f"KDA_OPTIMIZED_{kda_fn} = True\n")
                stamps.append(
                    f"KDA_TASK_{kda_fn} = {task_slug!r}\n"
                    f"KDA_COMMIT_{kda_fn} = {commit!r}\n"
                    f"KDA_DATE_{kda_fn} = {date!r}\n"
                    f"KDA_SPEEDUP_{kda_fn} = {speedup!r}\n"
                )
            else:
                imports.append(f"from {sgl_mod} import {kda_fn}  # noqa: F401\n")
                flags.append(f"KDA_OPTIMIZED_{kda_fn} = False\n")
        body = "".join(lines + imports + ["\n"] + flags + ["\n"] + stamps)
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


def cmd_list() -> int:
    for slug in sorted(
        p.name for p in KERNELS.iterdir() if p.is_dir() and "_diffusion_" in p.name
    ):
        impl = IMPLS / slug
        marker = " [exported]" if impl.exists() else ""
        print(f"  {slug}{marker}")
    return 0


def cmd_revert(task_slug: str) -> int:
    family = family_of(task_slug)
    impl = IMPLS / task_slug
    if impl.exists():
        shutil.rmtree(impl)
        print(f"removed kda_kernels/_impls/{task_slug}")
    regenerate_stub(family, task_slug, exports=set())
    print(f"reset kda_kernels stubs for family={family}")
    return 0


def cmd_export(task_slug: str) -> int:
    task_dir = KERNELS / task_slug
    if not task_dir.exists():
        raise SystemExit(f"no such task: {task_dir}")
    family = family_of(task_slug)
    src_dir = task_dir / "src"
    if not src_dir.exists():
        raise SystemExit(f"task {task_slug} has no src/")
    print(f"== exporting task={task_slug} family={family} ==")
    copy_src(task_dir, task_slug)
    exports = read_exports(src_dir)
    if not exports:
        print(
            "warning: src/register.py has no EXPORTS dict; "
            "stubs will fall back to sglang baseline",
            file=sys.stderr,
        )
    written = regenerate_stub(family, task_slug, exports)
    for p in written:
        print(f"  updated {p.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("task_slug", nargs="?", help="kernels/<slug> to promote")
    g.add_argument("--list", action="store_true")
    g.add_argument("--revert", metavar="TASK_SLUG", help="undo a promotion")
    args = ap.parse_args()
    if args.list:
        return cmd_list()
    if args.revert:
        return cmd_revert(args.revert)
    return cmd_export(args.task_slug)


if __name__ == "__main__":
    raise SystemExit(main())
