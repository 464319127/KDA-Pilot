# Profile evidence — glm52_bs1__silu_mul_quant_fp8_bitwise

silu_and_mul (~3.1 µs) + per_token_group_quant_fp8 (~1.7 µs) per MoE layer
at M=8, x75 layers ≈ 0.36 ms + in-graph launch gaps. Bitwise-identical
fusion is the only main-model-deployable form (accept numerics lock; see
mega_kernels/README.md rule 3). Precedent: tiny_align (single-launch align,
bitwise-identical combine) landed 289.9 -> 292.6 tok/s.
