"""Patch each generated task prompt to reference the empirical capture files.

Replaces the existing "Shape collection methodology" paragraph with a paragraph
that points to the right file names and explicitly lists which presets
contributed captures, falling back gracefully when a kernel has no captures.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
KERNELS_DIR = REPO_ROOT / "kernels"

FAMILIES = [
    "diffusion_qknorm_rope__multi_shape",
    "diffusion_rms_norm_fn__multi_shape",
    "diffusion_norm_infer__multi_shape",
    "diffusion_group_norm_silu__multi_shape",
    "diffusion_rotary_embedding__multi_shape",
    "diffusion_fuse_scale_shift__multi_shape",
    "diffusion_cutedsl_norm_tanh_mul_add__multi_shape",
    "diffusion_cutedsl_norm_scale_shift__multi_shape",
]
ARCHS = ["b200", "h200"]


def summarize_captures(folder: Path) -> tuple[set[str], int]:
    """Return (models_seen, unique_signatures) for a task folder."""
    models = set()
    signatures = set()
    for arch in ("b200", "h200"):
        p = folder / "docs" / f"captured_shapes_{arch}.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("model"):
                    models.add(rec["model"])
                # crude shape sig
                args = rec.get("args") or []
                kwargs = rec.get("kwargs") or {}
                key = json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str)
                signatures.add(key)
    return models, len(signatures)


def patch_prompt(folder: Path, models: set[str], n_sigs: int, arch: str) -> None:
    prompt_path = folder / "prompt.md"
    text = prompt_path.read_text()
    old_marker = "Shape collection methodology: the SGLang diffusion benchmark skill at"
    if old_marker not in text:
        return
    captured_arch = arch if (folder / "docs" / f"captured_shapes_{arch}.jsonl").stat().st_size > 0 else "h200"
    if not models:
        block = (
            "Shape collection methodology: the SGLang diffusion benchmark skill at\n"
            "`~/.codex/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py`\n"
            "was run for each preset with the `kernel_shape_capture.py` monkey-patch\n"
            "active on `ion-b200` (B200) and `ion8-h200` / `ion9-h200` (H200). For this\n"
            f"kernel family no live captures were observed in the latest sweep, so the\n"
            "table above reflects analytical/derived shapes from each model's published\n"
            "config. Re-run the sweep with the matching presets before final promotion;\n"
            "write the captured raw JSONL to `docs/captured_shapes_<arch>.jsonl`."
        )
    else:
        sorted_models = sorted(models)
        block = (
            "Shape collection methodology: the SGLang diffusion benchmark skill at\n"
            "`~/.codex/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py`\n"
            "was run for each preset with the `kernel_shape_capture.py` monkey-patch\n"
            "active on `ion-b200` (B200) and `ion8-h200` / `ion9-h200` (H200). For\n"
            f"this kernel family live captures fired on presets `{sorted_models}` and "
            f"are saved verbatim under `docs/captured_shapes_{captured_arch}.jsonl` and\n"
            f"summarized in `docs/captured_shapes_{captured_arch}.md` ({n_sigs} unique\n"
            "shape signatures). The analytical table above is the superset; any\n"
            "additional shape observed in a future capture must be appended before\n"
            "being claimed as part of the promotion target. Note: tensor shapes are\n"
            "arch-independent for this kernel; if `captured_shapes_b200.jsonl` is empty\n"
            "the agent must treat the H200 capture as the authoritative shape ledger."
        )

    end_marker = "table above before being claimed as part of the promotion target."
    start = text.index(old_marker)
    end = text.index(end_marker) + len(end_marker)
    new_text = text[:start] + block + text[end:]
    prompt_path.write_text(new_text)


def main() -> None:
    for family in FAMILIES:
        for arch in ARCHS:
            slug = f"{arch}_{family}"
            folder = KERNELS_DIR / slug
            if not folder.exists():
                continue
            models, n_sigs = summarize_captures(folder)
            patch_prompt(folder, models, n_sigs, arch)
            print(f"patched {slug}: models={sorted(models)} sigs={n_sigs}")


if __name__ == "__main__":
    main()
