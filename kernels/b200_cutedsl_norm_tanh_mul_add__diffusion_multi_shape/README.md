# b200_cutedsl_norm_tanh_mul_add__diffusion_multi_shape

KDA-style task folder for optimizing the SGLang diffusion
**CuTe-DSL norm + tanh(scale) + shift (+ second-norm scale)** kernel(s) on NVIDIA B200.

See `prompt.md` for the task contract, `interface.md` for the expected local
candidate interface, and `tests/test_correctness.py` for the correctness
oracle scaffold.

Wrapped SGLang baseline entry points:
- `sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale:fused_norm_tanh_mul_add`
- `sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale:fused_norm_tanh_mul_add_norm_scale`

Reference SGLang test used as the correctness oracle:
`python/sglang/jit_kernel/tests/diffusion/test_fused_norm_scale_shift.py`

Promotion target: at least 1.5x geometric-mean speedup over the SGLang
baseline across all configured shape buckets.
