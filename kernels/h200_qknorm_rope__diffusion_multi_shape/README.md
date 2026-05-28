# h200_qknorm_rope__diffusion_multi_shape

KDA-style task folder for optimizing the SGLang diffusion
**Fused QKNorm + RoPE (CUDA, in-place)** kernel(s) on NVIDIA H200.

See `prompt.md` for the task contract, `interface.md` for the expected local
candidate interface, and `tests/test_correctness.py` for the correctness
oracle scaffold.

Wrapped SGLang baseline entry points:
- `sglang.jit_kernel.diffusion.qknorm_rope:fused_inplace_qknorm_rope`

Reference SGLang test used as the correctness oracle:
`python/sglang/jit_kernel/tests/diffusion/test_qknorm_rope.py`

Promotion target: at least 1.3x geometric-mean speedup over the SGLang
baseline across all configured shape buckets.
