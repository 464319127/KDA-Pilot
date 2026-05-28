"""Registration stub for the B200 int8_scaled_mm KDA task.

Agents should replace `optimized_wrapper` with the recovered baseline-compatible
candidate wrapper during implementation.
"""

from __future__ import annotations

from typing import Any


KERNEL_SLUG = "b200_int8_scaled_mm__m64_n2048_k2048_bias"
OP_TYPE = "int8_scaled_mm"


def optimized_wrapper(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "Fill optimized_wrapper after recovering the SGLang int8_scaled_mm callsite."
    )


def register() -> dict[str, Any]:
    return {
        "name": KERNEL_SLUG,
        "op_type": OP_TYPE,
        "callable": optimized_wrapper,
        "version": "dev",
        "source": __file__,
    }
