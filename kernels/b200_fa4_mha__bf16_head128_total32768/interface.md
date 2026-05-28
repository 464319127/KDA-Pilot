# Interface: b200_fa4_mha__bf16_head128_total32768

- Kernel slug: `b200_fa4_mha__bf16_head128_total32768`
- Op type: `mha_forward`
- dtype: BF16
- head_dim: 128
- num_heads: 16
- total tokens: 32768
- Target GPU: NVIDIA B200

## Export

Provide:

```text
src/register.py
```

with:

```python
KERNEL_SLUG = "b200_fa4_mha__bf16_head128_total32768"
OP_TYPE = "mha_forward"

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

`optimized_wrapper` must preserve the recovered benchmark/harness contract for
BF16 Q/K/V MHA forward. It must fall back to the baseline implementation for
unsupported shapes, dtypes, layouts, devices, head dimensions, number of heads,
or causal modes.

The exact public signature should be filled after baseline recovery. A typical
shape-specialized wrapper may accept Q, K, V, causal flag, optional scale, and
workspace/stream metadata, but the final signature must match the harness used
for correctness and benchmark promotion.

## Evidence Requirements

Before promotion, update this file with:

- final wrapper signature;
- supported `(batch, seqlen, causal)` dispatch table;
- fallback cases;
- PyTorch/FP32 oracle tolerance methodology;
- benchmark command and TFLOPS formula;
- source lineage for copied or ported helper code.
