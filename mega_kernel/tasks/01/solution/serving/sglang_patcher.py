#!/usr/bin/env python3
"""Apply/revert the env-gated jit-port route in the serving checkout.

Exact-string patcher (idempotent, verifiable, reversible) for
python/sglang/srt/layers/flashinfer_comm_fusion.py in /sgl-workspace/sglang.
Deliberately NOT `git apply`: the serving worktree carries the production
patch (main_port_full.diff) as uncommitted changes, so all edits here must be
strictly additive and reversible by exact reverse replacement. NEVER run
`git checkout --` on that repo.

Usage:
  python3 sglang_patcher.py apply   [--repo /sgl-workspace/sglang]
  python3 sglang_patcher.py revert  [--repo /sgl-workspace/sglang]
  python3 sglang_patcher.py status  [--repo /sgl-workspace/sglang]

`apply` also copies the jit_kernel module files (mnnvl_ar_fused.py + csrc/)
into the checkout; `revert` removes them.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent

TARGET_REL = "python/sglang/srt/layers/flashinfer_comm_fusion.py"
JK_PY_REL = "python/sglang/jit_kernel/mnnvl_ar_fused.py"
JK_CSRC_REL = "python/sglang/jit_kernel/csrc/mnnvl_ar_fused"

# --- anchor: the output allocation right before the stock unified call ------
ANCHOR = """    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    kwargs = dict(
"""

ROUTED = """    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    # Task-local env-gated route (SGLANG_JIT_MNNVL_AR=1, default OFF): dispatch
    # the oneshot mnnvl decode regime to the jit_kernel port of the same
    # flashinfer kernel; every other case (twoshot/prefill sizes, fp32_acc,
    # non-bf16 dtype, trigger_completion_at_end, non-mnnvl backend, missing
    # workspace attrs) stays on the stock path so correctness is never lost.
    if _jit_mnnvl_ar_route(
        workspace_manager.workspace, input_tensor, use_oneshot, fp32_acc,
        use_attn_tp_group, trigger_completion_at_end,
    ):
        from sglang.jit_kernel.mnnvl_ar_fused import allreduce_add_rmsnorm

        allreduce_add_rmsnorm(
            input_tensor,
            workspace_manager.workspace,
            norm_out,
            residual_out,
            residual,
            weight,
            eps,
            launch_with_pdl=True,
        )
        return norm_out, residual_out

    kwargs = dict(
"""

HELPER_ANCHOR = """def fake_flashinfer_allreduce_residual_rmsnorm(
"""

HELPER_VERSION = "v3-trigger-guard"

HELPER = """def _jit_mnnvl_ar_route(ws, input_tensor, use_oneshot, fp32_acc, is_attn,
                        trigger_completion_at_end=False) -> bool:
    \"\"\"True only for the env-gated oneshot mnnvl decode regime (task 01; v3-trigger-guard).\"\"\"
    import os

    import torch

    if os.environ.get("SGLANG_JIT_MNNVL_AR", "0") != "1":
        return False
    scope = os.environ.get("SGLANG_JIT_MNNVL_AR_SCOPE", "all")
    if scope == "attn" and not is_attn:
        return False
    if scope == "moe" and is_attn:
        return False
    try:
        if ws is None or getattr(ws, "backend", None) != "mnnvl":
            return False
        if fp32_acc or use_oneshot is False:
            return False
        # The jit wrapper has no end-of-kernel completion parameter (the
        # kernel triggers PDL completion before the norm/output writes);
        # the stock path forwards this flag, so honor it there.
        if trigger_completion_at_end:
            return False
        # The jit module is bf16-only (it asserts); the stock flashinfer path
        # handles fp16/fp32, so any other dtype must stay on the stock route.
        if input_tensor.dtype != torch.bfloat16:
            return False
        num_tokens, hidden = input_tensor.shape
        payload = num_tokens * hidden * ws.tp_size * input_tensor.element_size()
        if payload > 64 * 1024 * 8 * 2:  # flashinfer MNNVL_ONE_SHOT_THRESHOLD
            return False
        for attr in (
            "mc_ptr",
            "uc_ptrs_dev",
            "uc_ptr_local",
            "buffer_flags",
            "tp_size",
            "rank",
        ):
            if not hasattr(ws, attr):
                return False
        return True
    except Exception:
        return False


def fake_flashinfer_allreduce_residual_rmsnorm(
"""


def target(repo: pathlib.Path) -> pathlib.Path:
    return repo / TARGET_REL


def is_applied(text: str) -> bool:
    return "_jit_mnnvl_ar_route" in text


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _copy_module_files(repo: pathlib.Path) -> None:
    """Always refresh the jit_kernel module files (idempotent).

    csrc comes DIRECTLY from solution/mnnvl_ar_fused/csrc — the single source
    of truth — never from a staged copy (a stale staged copy once shipped a
    module without the specialized symbol into a serving run).
    """
    shutil.copy2(HERE / "files" / "mnnvl_ar_fused.py", repo / JK_PY_REL)
    csrc_src = HERE.parent / "mnnvl_ar_fused" / "csrc"
    csrc_dst = repo / JK_CSRC_REL
    csrc_dst.mkdir(parents=True, exist_ok=True)
    for f in csrc_src.iterdir():
        if f.suffix in (".cu", ".cuh"):
            shutil.copy2(f, csrc_dst / f.name)


def apply(repo: pathlib.Path) -> int:
    t = target(repo)
    text = t.read_text()
    if is_applied(text):
        if HELPER not in text:
            # The route exists but its text differs from this patcher's HELPER
            # (an older helper version is deployed). Refusing beats silently
            # keeping stale route logic in the checkout.
            print(f"ERROR: _jit_mnnvl_ar_route is applied with a DIFFERENT helper "
                  f"version than this patcher ({HELPER_VERSION}). Revert with the "
                  f"patcher version that applied it, then re-run apply.")
            return 1
        # comm_fusion already routed at the current version — still refresh the
        # module files so flag or source updates always land (an early return
        # here once shipped a stale build into a full serving gate run).
        _copy_module_files(repo)
        print(f"already applied at {HELPER_VERSION} ({TARGET_REL} sha={sha(t)}); "
              f"module files refreshed")
        return 0
    if text.count(ANCHOR) != 1 or text.count(HELPER_ANCHOR) != 1:
        print("ERROR: anchors not found exactly once — serving file drifted; aborting")
        return 1
    before = sha(t)
    text = text.replace(ANCHOR, ROUTED).replace(HELPER_ANCHOR, HELPER)
    t.write_text(text)
    _copy_module_files(repo)
    print(f"applied: {TARGET_REL} {before} -> {sha(t)}; "
          f"added {JK_PY_REL}, {JK_CSRC_REL}/")
    return 0


def revert(repo: pathlib.Path) -> int:
    t = target(repo)
    text = t.read_text()
    if not is_applied(text):
        print("not applied")
    else:
        if text.count(ROUTED) != 1 or text.count(HELPER) != 1:
            print("ERROR: applied blocks not found exactly once; manual review needed")
            return 1
        before = sha(t)
        text = text.replace(ROUTED, ANCHOR).replace(HELPER, HELPER_ANCHOR)
        t.write_text(text)
        print(f"reverted: {TARGET_REL} {before} -> {sha(t)}")
    py = repo / JK_PY_REL
    if py.exists():
        py.unlink()
        print(f"removed {JK_PY_REL}")
    csrc = repo / JK_CSRC_REL
    if csrc.exists():
        shutil.rmtree(csrc)
        print(f"removed {JK_CSRC_REL}/")
    return 0


def status(repo: pathlib.Path) -> int:
    t = target(repo)
    print(f"{TARGET_REL}: sha={sha(t)} applied={is_applied(t.read_text())}")
    print(f"{JK_PY_REL}: exists={(repo / JK_PY_REL).exists()}")
    print(f"{JK_CSRC_REL}/: exists={(repo / JK_CSRC_REL).exists()}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["apply", "revert", "status"])
    ap.add_argument("--repo", default="/sgl-workspace/sglang")
    a = ap.parse_args()
    sys.exit({"apply": apply, "revert": revert, "status": status}[a.action](
        pathlib.Path(a.repo)))
