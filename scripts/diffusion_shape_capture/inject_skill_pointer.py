"""Inject an explicit `## Required Claude Code Skill` section into each
diffusion task prompt, so the agent that picks the task up doesn't have to
infer the local skill dependency from the host alias.

For B200 tasks the skill is `ion-b200`. For H200 tasks the skill is
`ion8-h200` (primary) or `ion9-h200` (interchangeable backup); both expose the
same H200 container topology.

The section is inserted right before the existing `## Environment And Remote
Rule` heading, and replaces itself idempotently on rerun.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent
KERNELS_DIR = REPO / "kernels"

ANCHOR = "## Environment And Remote Rule"
SECTION_HEADER = "## Required Claude Code Skill"


def render_section(arch: str) -> str:
    if arch == "b200":
        lines = [
            SECTION_HEADER,
            "",
            "This task talks to the remote GPU box exclusively through the local",
            "Claude Code skill `~/.claude/skills/ion-b200/SKILL.md`. The skill",
            "owns the SSH alias `ion-b200`, the `sglang_bbuf` Docker container",
            "lifecycle (the create command keeps `--privileged --cap-add=SYS_ADMIN",
            "--security-opt seccomp=unconfined` so `ncu --set basic` can collect",
            "counters), the idle-GPU selection rule (`nvidia-smi` 0% util + low",
            "memory), and the `kill-idle` shortcut.",
            "",
            "If the skill is missing on the box that launches Humanize/RLCR, fetch",
            "it before starting the loop; do not paraphrase the SSH pattern by",
            "hand. The skill's SKILL.md is the single source of truth for `ion-b200`",
            "host conventions; this prompt only consumes them.",
            "",
        ]
    else:
        lines = [
            SECTION_HEADER,
            "",
            "This task talks to the remote GPU box exclusively through the local",
            "Claude Code skills `~/.claude/skills/ion8-h200/SKILL.md` (primary)",
            "and `~/.claude/skills/ion9-h200/SKILL.md` (interchangeable backup",
            "with the same H200 topology and the same `sglang_bbuf` container).",
            "Either skill owns the SSH alias, container lifecycle (privileged",
            "+ Nsight Compute access), idle-GPU selection rule, and the",
            "`kill-idle` shortcut.",
            "",
            "Pick whichever skill currently has idle H200 GPUs available; the",
            "remote workspace path, container name, and benchmark commands in",
            "the next section apply to both. If neither skill is loaded, fetch",
            "them before starting the loop; do not paraphrase the SSH pattern by",
            "hand.",
            "",
        ]
    return "\n".join(lines)


def patch_one(prompt_path: Path, arch: str) -> None:
    text = prompt_path.read_text(encoding="utf-8")
    section = render_section(arch)

    if SECTION_HEADER in text:
        start = text.index(SECTION_HEADER)
        rest = text[start:]
        next_header_idx = rest.find("\n## ", len(SECTION_HEADER))
        if next_header_idx == -1:
            new_text = text[:start] + section.rstrip() + "\n"
        else:
            new_text = text[:start] + section.rstrip() + "\n\n" + rest[next_header_idx + 1 :]
    else:
        anchor_idx = text.find(ANCHOR)
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


def main() -> None:
    for task_dir in sorted(KERNELS_DIR.iterdir()):
        if not task_dir.is_dir() or "diffusion_" not in task_dir.name:
            continue
        prompt = task_dir / "prompt.md"
        if not prompt.exists():
            continue
        arch = "b200" if task_dir.name.startswith("b200_") else "h200"
        patch_one(prompt, arch)
        print(f"patched {task_dir.name}")


if __name__ == "__main__":
    main()
