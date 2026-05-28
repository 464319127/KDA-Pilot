# h200_diffusion_rotary_embedding__multi_shape

KDA-style task folder for optimizing the SGLang diffusion
**Standard and LTX-2 split rotary embeddings** kernel(s) on NVIDIA H200.

See `prompt.md` for the task contract, `interface.md` for the expected local
candidate interface, and `tests/test_correctness.py` for the correctness
oracle scaffold.

Wrapped SGLang baseline entry points:
- `sglang.jit_kernel.diffusion.triton.rotary:apply_rotary_embedding`
- `sglang.jit_kernel.diffusion.triton.ltx2_rotary:apply_ltx2_split_rotary_emb`

Reference SGLang test used as the correctness oracle:
`python/sglang/jit_kernel/tests/test_rope.py`

Promotion target: at least 1.4x geometric-mean speedup over the SGLang
baseline across all configured shape buckets.
