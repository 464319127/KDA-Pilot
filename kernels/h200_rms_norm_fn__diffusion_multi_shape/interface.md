# Interface: h200_rms_norm_fn__diffusion_multi_shape

- Kernel slug: `h200_rms_norm_fn__diffusion_multi_shape`
- Op type: `layer_or_rms_norm_fn`
- Target GPU: NVIDIA H200
- Wrapped SGLang entry points:
- `sglang.jit_kernel.diffusion.triton.norm:rms_norm_fn`

## Export

Provide:

```text
src/register.py
```

with:

```python
KERNEL_SLUG = "h200_rms_norm_fn__diffusion_multi_shape"
OP_TYPE = "layer_or_rms_norm_fn"

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

`optimized_wrapper` must preserve the recovered SGLang callsite contract
for every wrapped entry point. It must fall back to the baseline
implementation for any shape, dtype, layout, device, normalization type,
or feature flag that is not part of the configured shape table.

The exact public signature for each wrapped entry point should be filled
after baseline recovery. Typical wrappers for this family accept the same
positional and keyword arguments as the SGLang baseline (see `prompt.md`),
plus optional `*, dispatcher_hint=` keyword for dispatcher overrides.

## Evidence Requirements

Before promotion, update this file with:

- final wrapper signature(s);
- per-shape dispatch table (which underlying candidate kernel handles
which shape bucket);
- fallback cases;
- PyTorch-FP32 or `_reference()` tolerance methodology used in tests;
- benchmark command and latency formula;
- source lineage for copied or ported helper code.
