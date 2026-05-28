# b200_rms_norm_fn__diffusion_multi_shape

KDA-style task folder for optimizing the SGLang diffusion
**Flash-attn-style 1-pass LN/RMSN with optional residual** kernel(s) on NVIDIA B200.

See `prompt.md` for the task contract, `interface.md` for the expected local
candidate interface, and `tests/test_correctness.py` for the correctness
oracle scaffold.

Wrapped SGLang baseline entry points:
- `sglang.jit_kernel.diffusion.triton.norm:rms_norm_fn`

Reference SGLang test used as the correctness oracle:
`python/sglang/jit_kernel/tests/test_rmsnorm.py`

Promotion target: at least 1.5x geometric-mean speedup over the SGLang
baseline across all configured shape buckets.
