"""Runtime monkey-patch of sglang.jit_kernel.diffusion.* with kda_kernels."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from kda_kernels._registry import KERNEL_REGISTRY

_log = logging.getLogger("kda_kernels.install")
_installed: dict[str, tuple[Any, str, Any]] = {}


def install(force: bool = False, strict: bool = False) -> list[tuple[str, str, str]]:
    """Swap KDA-optimized diffusion kernels into sglang at runtime.

    For each (sglang_path, kda_path) pair in
    `kda_kernels._registry.KERNEL_REGISTRY`, this function:

    1. Imports the sglang side and the kda_kernels side.
    2. Checks `getattr(kda_module, f"KDA_OPTIMIZED_{kda_name}", False)`.
       If False, the function is left untouched (the kda stub still
       re-exports the sglang baseline anyway, so calling install() with
       nothing exported is harmless).
    3. If the module exposes `_preload_kda_impls`, imports promoted arch
       wrappers before monkey-patching so their baseline fallbacks capture the
       original SGLang functions.
    4. If True, replaces the attribute on the sglang module with the
       kda_kernels version. The original is saved so `uninstall()` can
       restore it.

    Args:
      force: if True, re-swap functions even when they are already
        marked installed.
      strict: if True, raise on the first failure instead of skipping.

    Returns:
      List of (sglang_path, kda_path, status) tuples. Statuses:
        - "swapped": kda version is now bound to the sglang attribute.
        - "skipped: not optimized": kda stub still re-exports baseline.
        - "skipped: already installed": call again with force=True to
          re-swap.
        - "skipped: <error>": import or attribute access failed.
    """
    results: list[tuple[str, str, str]] = []
    for sglang_path, kda_path in KERNEL_REGISTRY.items():
        if (not force) and sglang_path in _installed:
            results.append((sglang_path, kda_path, "skipped: already installed"))
            continue
        sgl_mod, sgl_name = sglang_path.split(":")
        kda_mod, kda_name = kda_path.split(":")
        try:
            kda_module = importlib.import_module(kda_mod)
            flag = getattr(kda_module, f"KDA_OPTIMIZED_{kda_name}", False)
            if not flag:
                results.append((sglang_path, kda_path, "skipped: not optimized"))
                continue
            preload = getattr(kda_module, "_preload_kda_impls", None)
            if callable(preload):
                preload(strict=strict)
            sgl_module = importlib.import_module(sgl_mod)
            original = getattr(sgl_module, sgl_name)
            replacement = getattr(kda_module, kda_name)
            _installed[sglang_path] = (sgl_module, sgl_name, original)
            setattr(sgl_module, sgl_name, replacement)
            results.append((sglang_path, kda_path, "swapped"))
            _log.info("kda_kernels swapped %s -> %s", sglang_path, kda_path)
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            results.append((sglang_path, kda_path, f"skipped: {exc!r}"))
            _log.warning("kda_kernels skip %s -> %s: %r", sglang_path, kda_path, exc)
    return results


def uninstall() -> list[str]:
    """Restore the original sglang baseline functions. Returns the list
    of sglang_path entries that were restored.
    """
    restored: list[str] = []
    for sglang_path, (sgl_module, sgl_name, original) in list(_installed.items()):
        setattr(sgl_module, sgl_name, original)
        _installed.pop(sglang_path)
        restored.append(sglang_path)
    return restored


def status() -> list[str]:
    """Return the list of sglang_path entries currently installed."""
    return sorted(_installed.keys())
