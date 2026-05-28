"""Inject a `## External Reference Skills` section into every diffusion task
prompt so the optimizing agent has concrete KernelWiki query examples and a
concrete ncu-report-skill workflow, scoped to that kernel family.

The section lands immediately after `## Prior Art Research Scope` (where
KernelWiki is mentioned generically) and replaces itself idempotently.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent
KERNELS_DIR = REPO / "kernels"

ANCHOR_END = "## Optimization Exploration Policy"
SECTION_HEADER = "## External Reference Skills"

# Per-family KernelWiki query suggestions. Keys are the family slug
# (after stripping `{arch}_diffusion_` and `__multi_shape`).
KW_QUERIES: dict[str, list[str]] = {
    "qknorm_rope": [
        'python3 ../../external/KernelWiki/scripts/query.py "fused RMS norm + RoPE inplace sm100"',
        'python3 ../../external/KernelWiki/scripts/query.py "qk norm rope blackwell"',
        'python3 ../../external/KernelWiki/scripts/query.py --tag qk-norm --type kernel',
        'python3 ../../external/KernelWiki/scripts/query.py --repo sglang --tag rope --architecture sm100 --limit 20',
        'python3 ../../external/KernelWiki/scripts/query.py --repo flashinfer --tag rope --limit 20',
    ],
    "rms_norm_fn": [
        'python3 ../../external/KernelWiki/scripts/query.py "flash-attention layer norm residual fused"',
        'python3 ../../external/KernelWiki/scripts/query.py --tag layer-norm --type kernel',
        'python3 ../../external/KernelWiki/scripts/query.py --tag rms-norm --architecture sm100 --limit 20',
        'python3 ../../external/KernelWiki/scripts/query.py --repo flash-attention --tag norm --limit 20',
    ],
    "norm_infer": [
        'python3 ../../external/KernelWiki/scripts/query.py "layer norm 2-pass inference sm100"',
        'python3 ../../external/KernelWiki/scripts/query.py "one pass rmsnorm tiled per head"',
        'python3 ../../external/KernelWiki/scripts/query.py --tag rms-norm --architecture sm100 --limit 20',
        'python3 ../../external/KernelWiki/scripts/query.py --tag layer-norm --architecture sm100 --limit 20',
    ],
    "group_norm_silu": [
        'python3 ../../external/KernelWiki/scripts/query.py "group norm silu fused vae blackwell"',
        'python3 ../../external/KernelWiki/scripts/query.py --tag group-norm --type kernel',
        'python3 ../../external/KernelWiki/scripts/query.py --repo pytorch --tag group-norm --limit 20',
        'python3 ../../external/KernelWiki/scripts/query.py --symptom memory-bound --tag norm --limit 20',
    ],
    "rotary_embedding": [
        'python3 ../../external/KernelWiki/scripts/query.py "rope interleaved neox sm100"',
        'python3 ../../external/KernelWiki/scripts/query.py --tag rope --type kernel',
        'python3 ../../external/KernelWiki/scripts/query.py --tag rope --architecture sm100 --limit 20',
        'python3 ../../external/KernelWiki/scripts/query.py --repo flashinfer --tag rope --limit 20',
    ],
    "fuse_scale_shift": [
        'python3 ../../external/KernelWiki/scripts/query.py "adaLN modulation fused scale shift gate"',
        'python3 ../../external/KernelWiki/scripts/query.py "DiT modulation kernel sm100"',
        'python3 ../../external/KernelWiki/scripts/query.py --tag modulation --type kernel',
        'python3 ../../external/KernelWiki/scripts/query.py --repo sglang --tag modulation --limit 20',
    ],
    "cutedsl_norm_tanh_mul_add": [
        'python3 ../../external/KernelWiki/scripts/query.py "CuTe DSL layer norm modulation sm100"',
        'python3 ../../external/KernelWiki/scripts/query.py "Z-Image residual modulation kernel"',
        'python3 ../../external/KernelWiki/scripts/query.py --tag cute-dsl --type kernel',
        'python3 ../../external/KernelWiki/scripts/query.py --tag modulation --architecture sm100 --limit 20',
    ],
    "cutedsl_norm_scale_shift": [
        'python3 ../../external/KernelWiki/scripts/query.py "CuTe DSL norm scale shift fused"',
        'python3 ../../external/KernelWiki/scripts/query.py "fused norm residual gate adaLN"',
        'python3 ../../external/KernelWiki/scripts/query.py --tag cute-dsl --type kernel',
        'python3 ../../external/KernelWiki/scripts/query.py --tag modulation --architecture sm100 --limit 20',
    ],
}

# Per-family one-liner reminder of what the family actually does, used in the
# "what to look for" sentence inside the injected section.
FAMILY_BLURB: dict[str, str] = {
    "qknorm_rope": "fused in-place RMS-norm + RoPE on (Q, K) for DiT joint attention blocks",
    "rms_norm_fn": "flash-attention-style 1-pass LayerNorm / RMSNorm with optional residual and dual-branch",
    "norm_infer": "inference-only 2-pass LayerNorm / RMSNorm and tiled one-pass RMSNorm baselines",
    "group_norm_silu": "fused GroupNorm + SiLU used by image (2D/3D) and video (5D) VAE decoders",
    "rotary_embedding": "standard apply_rotary_embedding plus the LTX-2 split-RoPE variant with 4D cos/sin",
    "fuse_scale_shift": "fused scale-shift modulation including the Z-Image / Qwen-Image-Edit `select01` dual-modulation path",
    "cutedsl_norm_tanh_mul_add": "Z-Image residual modulation `norm(x) * tanh(scale) + shift` (with the second-norm-scale variant)",
    "cutedsl_norm_scale_shift": "fused `norm(x) * (1 + scale) + shift` and its residual+gate variant used by Qwen-Image / Wan / HunyuanVideo / Helios",
}

# Whether the candidate is a native CUDA kernel (qknorm_rope) or a Triton /
# CuTe-DSL kernel — drives the ncu harness hint.
def is_cuda_family(family: str) -> bool:
    return family == "qknorm_rope"


def render_section(family: str) -> str:
    queries = KW_QUERIES[family]
    blurb = FAMILY_BLURB[family]
    bullets = "\n".join(f"  - `{q}`" for q in queries)
    harness_note = (
        "Because this family is a native CUDA kernel, build the harness from\n"
        "  your `src/` `.cu` / `.cuh` sources with `-lineinfo` so SASS maps back\n"
        "  to source. Match `head_dim` / `rope_dim` / `num_heads` / `is_neox`\n"
        "  exactly to the shape under investigation."
        if is_cuda_family(family)
        else "Because this family is a Triton or CuTe-DSL kernel, the harness\n"
        "  should `python -c` into the wrapped SGLang entry point with the\n"
        "  exact captured shape from `docs/captured_shapes_<arch>.jsonl`. The\n"
        "  generated Triton SASS lives under the Triton cache (`~/.triton/cache/`)\n"
        "  and can be passed to ncu via `--app-replay-mode kernel`."
    )
    return (
        f"{SECTION_HEADER}\n"
        f"\n"
        f"Two repo-vendored skills live under `../../external/` (they are git\n"
        f"submodules; if either folder is empty, run\n"
        f"`git submodule update --init --recursive` from the repo root before\n"
        f"starting RLCR).\n"
        f"\n"
        f"### KernelWiki (`../../external/KernelWiki/`)\n"
        f"\n"
        f"Searchable knowledge base of 2,179 merged PRs across CUTLASS, SGLang,\n"
        f"vLLM, FlashInfer, PyTorch, Triton (Blackwell / SM100 + Hopper / SM90),\n"
        f"plus 48 wiki pages, 7 competitions, and 20 blog summaries. Read\n"
        f"`../../external/KernelWiki/SKILL.md` first for the full query syntax.\n"
        f"\n"
        f"Use it **before** locking in an implementation direction, especially\n"
        f"when designing the first benchmark-targeting candidate or after one\n"
        f"focused attempt comes back > 3× slower than the SGLang baseline.\n"
        f"\n"
        f"Targeted queries for this kernel family ({blurb}):\n"
        f"\n"
        f"{bullets}\n"
        f"\n"
        f"Record any PR / wiki page that influenced a design decision in\n"
        f"`docs/draft.md` and `solutions.jsonl` with its KernelWiki page id\n"
        f"or upstream PR URL.\n"
        f"\n"
        f"### ncu-report-skill (`../../external/ncu-report-skill/`)\n"
        f"\n"
        f"Nsight Compute B200 / SM100 workflow. Read\n"
        f"`../../external/ncu-report-skill/SKILL.md` first; the\n"
        f"`reference/` subdirectory has the directory layout, harness guide,\n"
        f"collection options, Python API, six analysis dimensions, diagnosis\n"
        f"playbook, and report template.\n"
        f"\n"
        f"Use it **after** the candidate is correct on all configured shapes\n"
        f"and either (a) the benchmark is within ~2× of the baseline and you\n"
        f"need to identify the active hardware bound, or (b) an attempted\n"
        f"optimization gave an unexpected result. The mandatory pattern in\n"
        f"this repo:\n"
        f"\n"
        f"  1. Create `profile/<run_name>/{{harness,reports,analysis}}/` — one\n"
        f"     dir per run, never reuse.\n"
        f"  2. Build a standalone harness under `profile/<run_name>/harness/`.\n"
        f"     {harness_note}\n"
        f"  3. Run two profiles into `profile/<run_name>/reports/`:\n"
        f"\n"
        f"     ```bash\n"
        f"     ncu --set full --target-processes all \\\n"
        f"       -o profile/<run_name>/reports/full \\\n"
        f"       <harness binary or python entrypoint>\n"
        f"\n"
        f"     ncu --set source --section SourceCounters \\\n"
        f"       -o profile/<run_name>/reports/source \\\n"
        f"       <harness binary or python entrypoint>\n"
        f"     ```\n"
        f"\n"
        f"  4. Parse with the `ncu_report` Python module via the helpers in\n"
        f"     `../../external/ncu-report-skill/helpers/`; write parsed CSVs\n"
        f"     into `profile/<run_name>/analysis/`.\n"
        f"  5. Walk the six analysis dimensions (compute / memory / occupancy\n"
        f"     / latency-hiding / launch-overhead / tail-effect) listed in\n"
        f"     `../../external/ncu-report-skill/reference/05-analysis-dimensions.md`.\n"
        f"  6. Match the dominant signal to\n"
        f"     `../../external/ncu-report-skill/reference/06-diagnosis-playbook.md`\n"
        f"     and write `profile/<run_name>/REPORT.md` using\n"
        f"     `reference/07-report-template.md`.\n"
        f"\n"
        f"Record the matched diagnosis, before/after metric, and the resulting\n"
        f"design change in `solutions.jsonl` together with the report path.\n"
        f"\n"
    )


def patch_one(prompt_path: Path, family: str) -> None:
    text = prompt_path.read_text(encoding="utf-8")
    section = render_section(family)

    if SECTION_HEADER in text:
        start = text.index(SECTION_HEADER)
        rest = text[start:]
        next_header_idx = rest.find("\n## ", len(SECTION_HEADER))
        if next_header_idx == -1:
            new_text = text[:start] + section.rstrip() + "\n"
        else:
            new_text = text[:start] + section.rstrip() + "\n\n" + rest[next_header_idx + 1 :]
    else:
        anchor_idx = text.find(ANCHOR_END)
        if anchor_idx == -1:
            new_text = text.rstrip() + "\n\n" + section.rstrip() + "\n"
        else:
            new_text = (
                text[:anchor_idx].rstrip()
                + "\n\n"
                + section.rstrip()
                + "\n\n"
                + text[anchor_idx:]
            )
    prompt_path.write_text(new_text, encoding="utf-8")


def family_of(task_name: str) -> str:
    # task_name looks like `b200_diffusion_qknorm_rope__multi_shape`
    if not task_name.startswith(("b200_diffusion_", "h200_diffusion_")):
        return ""
    stripped = task_name.split("_diffusion_", 1)[1]
    return stripped.split("__multi_shape", 1)[0]


def main() -> None:
    for task_dir in sorted(KERNELS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        family = family_of(task_dir.name)
        if family not in KW_QUERIES:
            continue
        prompt = task_dir / "prompt.md"
        if not prompt.exists():
            continue
        patch_one(prompt, family)
        print(f"patched {task_dir.name} (family={family})")


if __name__ == "__main__":
    main()
