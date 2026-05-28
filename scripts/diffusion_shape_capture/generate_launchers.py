"""Generate per-task launcher shell scripts in scripts/launch_kernels/.

Naming follows the existing pattern `kNN_<task-slug>.sh`. The launcher delegates to
`scripts/launch_kda_kernel_task.sh` with the task folder relative path.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
LAUNCH_DIR = REPO_ROOT / "scripts" / "launch_kernels"

# Order matters: assigns k03, k04, ... starting after the existing two.
TASKS = [
    # (kernel-folder-name,)
    "b200_qknorm_rope__diffusion_multi_shape",
    "h200_qknorm_rope__diffusion_multi_shape",
    "b200_rms_norm_fn__diffusion_multi_shape",
    "h200_rms_norm_fn__diffusion_multi_shape",
    "b200_norm_infer__diffusion_multi_shape",
    "h200_norm_infer__diffusion_multi_shape",
    "b200_group_norm_silu__diffusion_multi_shape",
    "h200_group_norm_silu__diffusion_multi_shape",
    "b200_rotary_embedding__diffusion_multi_shape",
    "h200_rotary_embedding__diffusion_multi_shape",
    "b200_fuse_scale_shift__diffusion_multi_shape",
    "h200_fuse_scale_shift__diffusion_multi_shape",
    "b200_cutedsl_norm_tanh_mul_add__diffusion_multi_shape",
    "h200_cutedsl_norm_tanh_mul_add__diffusion_multi_shape",
    "b200_cutedsl_norm_scale_shift__diffusion_multi_shape",
    "h200_cutedsl_norm_scale_shift__diffusion_multi_shape",
]

LAUNCHER_TEMPLATE = """\
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
KDA_LAUNCHER_NAME="${{KDA_LAUNCHER_NAME:-$(basename "$0")}}"
export KDA_LAUNCHER_NAME
exec "$SCRIPT_DIR/../launch_kda_kernel_task.sh" "kernels/{task}" "$@"
"""


def main() -> None:
    LAUNCH_DIR.mkdir(parents=True, exist_ok=True)
    next_index = 3
    for task in TASKS:
        index = next_index
        next_index += 1
        path = LAUNCH_DIR / f"k{index:02d}_{task}.sh"
        path.write_text(LAUNCHER_TEMPLATE.format(task=task))
        path.chmod(0o755)
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
