"""Correctness scaffold for ``b200_diffusion_cutedsl_norm_scale_shift__multi_shape``.

This file is intentionally skipped unless ``KDA_RUN_CORRECTNESS=1`` is set.

Required agent edits:
- replace ``make_cases()`` with the full configured shape list from
  ``prompt.md``;
- implement ``baseline(case)`` by calling the wrapped SGLang baseline entry
  points listed in ``prompt.md`` (treat that as the semantic oracle for this
  task and cross-check against a PyTorch FP32 reference where practical);
- keep ``candidate(case)`` compatible with ``src/register.py``;
- use dynamic BF16/FP16-aware tolerances where applicable.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - CUDA env owns the real run
    torch = None


KERNEL_SLUG = "b200_diffusion_cutedsl_norm_scale_shift__multi_shape"
OP_TYPE = "cutedsl_norm_scale_shift"
KERNEL_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.environ.get("KDA_RUN_CORRECTNESS") != "1",
    reason="Set KDA_RUN_CORRECTNESS=1 after the SGLang baseline cases are filled.",
)


def _load_register_module():
    register_py = KERNEL_DIR / "src" / "register.py"
    spec = importlib.util.spec_from_file_location(
        f"kda_kernel_{KERNEL_SLUG}_register", register_py
    )
    assert spec is not None and spec.loader is not None, register_py
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_cases() -> list[dict[str, Any]]:
    """Return all configured correctness/benchmark cases.

    The list must cover every shape bucket recorded in ``prompt.md``. Each case
    typically looks like::

        {
            "name": "flux__bf16__B1S4608D3072",
            "model": "flux",
            "args": (...),
            "kwargs": {...},
            "atol": 5e-2,
            "rtol": 5e-2,
            "warmup": 25,
            "iters": 100,
        }

    Returning an empty list keeps the scaffold skipped.
    """

    return []


def baseline(case: dict[str, Any]) -> Any:
    """Return the SGLang baseline result for one configured case."""

    raise NotImplementedError(
        "Call the SGLang baseline entry point(s) listed in prompt.md as the oracle."
    )


def candidate(case: dict[str, Any]) -> Any:
    module = _load_register_module()
    wrapper = getattr(module, "optimized_wrapper")
    args = case.get("args", ())
    kwargs = case.get("kwargs", {})
    return wrapper(*args, **kwargs)


def _assert_no_nan_inf(value: Any, *, path: str) -> None:
    if torch is not None and isinstance(value, torch.Tensor):
        assert not torch.isnan(value).any(), f"{path} contains NaN"
        assert not torch.isinf(value).any(), f"{path} contains Inf"
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            _assert_no_nan_inf(item, path=f"{path}[{i}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_no_nan_inf(item, path=f"{path}.{key}")


def _assert_close(actual: Any, expected: Any, *, case: dict[str, Any], path: str = "out") -> None:
    atol = case.get("atol", 5e-2)
    rtol = case.get("rtol", 5e-2)
    _assert_no_nan_inf(actual, path=path)
    if torch is not None and isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor), f"{path} expected tensor, got {type(expected)}"
        assert actual.shape == expected.shape, f"{path} shape {actual.shape} != {expected.shape}"
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
        return
    if isinstance(actual, (tuple, list)):
        assert isinstance(expected, type(actual)), f"{path} type mismatch"
        assert len(actual) == len(expected), f"{path} length mismatch"
        for i, (a_item, e_item) in enumerate(zip(actual, expected)):
            _assert_close(a_item, e_item, case=case, path=f"{path}[{i}]")
        return
    if isinstance(actual, dict):
        assert isinstance(expected, dict), f"{path} expected dict"
        assert actual.keys() == expected.keys(), f"{path} keys mismatch"
        for key in actual:
            _assert_close(actual[key], expected[key], case=case, path=f"{path}.{key}")
        return
    assert actual == expected, f"{path} value mismatch"


def test_register_metadata() -> None:
    module = _load_register_module()
    assert hasattr(module, "register")
    spec = module.register()
    assert spec["name"] == KERNEL_SLUG
    assert spec["op_type"] == OP_TYPE
    assert callable(spec["callable"])


def test_correctness_cases() -> None:
    cases = make_cases()
    assert cases, "No correctness cases recovered. Fill make_cases() before optimizing."
    for case in cases:
        expected = baseline(case)
        actual = candidate(case)
        _assert_close(actual, expected, case=case, path=case.get("name", "out"))
