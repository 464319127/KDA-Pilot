# Interface: b200_int8_scaled_mm__m64_n2048_k2048_bias

- Kernel slug: `b200_int8_scaled_mm__m64_n2048_k2048_bias`
- Op type: `int8_scaled_mm`
- Target shape: `M=64, N=2048, K=2048`
- Output dtype: `fp16`
- Bias: required
- Target GPU: NVIDIA B200

## Export

Provide:

```text
src/register.py
```

with:

```python
KERNEL_SLUG = "b200_int8_scaled_mm__m64_n2048_k2048_bias"
OP_TYPE = "int8_scaled_mm"

def optimized_wrapper(*args, **kwargs):
    ...

def register() -> dict:
    return {
        "name": KERNEL_SLUG,
        "op_type": OP_TYPE,
        "callable": optimized_wrapper,
        "version": "dev",
        "source": __file__,
    }
```

`optimized_wrapper` must preserve the recovered SGLang `int8_scaled_mm`
callsite contract. If the input shape, dtype, layout, device, scale format, or
bias mode is unsupported, it must fall back to the baseline implementation
rather than silently returning an approximate result.

The exact public signature should be filled after baseline recovery. Hide any
candidate-specific argument reshaping inside the wrapper so callers can keep the
original SGLang behavior.

## Evidence Requirements

Before promotion, update this file with:

- final wrapper signature;
- supported shape/dtype/layout matrix;
- fallback cases;
- correctness tolerance;
- benchmark command;
- source lineage for copied or ported helper code.
