"""Correctness scaffold for `b200_fa4_mha__bf16_head128_total32768`.

This file is intentionally skipped unless `KDA_RUN_CORRECTNESS=1` is set.

Required agent edits:
- replace `make_cases()` with all configured `(batch, seqlen, causal)` cases;
- implement `baseline(case)` with the semantic PyTorch/FP32 oracle and optional
  FA4 diagnostic output;
- keep `candidate(case)` compatible with `src/register.py`;
- use dynamic BF16-aware tolerances where practical.
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


KERNEL_SLUG = "b200_fa4_mha__bf16_head128_total32768"
OP_TYPE = "mha_forward"
KERNEL_DIR = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.environ.get("KDA_RUN_CORRECTNESS") != "1",
    reason="Set KDA_RUN_CORRECTNESS=1 after MHA baseline cases are filled.",
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
    """Return all configured BF16 MHA benchmark/correctness cases."""

    return []


def baseline(case: dict[str, Any]) -> Any:
    """Return the PyTorch/FP32 semantic oracle for one recovered case."""

    raise NotImplementedError("Fill baseline(case) with the PyTorch/FP32 oracle.")


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
    atol = case.get("atol", 2e-2)
    rtol = case.get("rtol", 1e-1)
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
