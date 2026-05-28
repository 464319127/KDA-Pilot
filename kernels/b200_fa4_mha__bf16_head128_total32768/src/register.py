"""Registration stub for the B200 FA4-comparable MHA KDA task.

Agents should replace `optimized_wrapper` with the recovered benchmark-compatible
candidate wrapper during implementation.
"""

from __future__ import annotations

from typing import Any


KERNEL_SLUG = "b200_fa4_mha__bf16_head128_total32768"
OP_TYPE = "mha_forward"


def optimized_wrapper(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "Fill optimized_wrapper after recovering the BF16 MHA benchmark contract."
    )


def register() -> dict[str, Any]:
    return {
        "name": KERNEL_SLUG,
        "op_type": OP_TYPE,
        "callable": optimized_wrapper,
        "version": "dev",
        "source": __file__,
    }
