# Profile evidence — glm52_bs1__oneshot_allreduce_bf16

~156 all-reduces per decode round (2/layer x 78: post-o_proj, post-MoE) of
bf16 [M<=8, 6144], all inside CUDA graphs. NCCL NVLS measured 12.1 µs
in-graph per call on this fleet => ~1.9 ms/round (~11%). vLLM-style LD/ST
one-shot measured 20.4-32 µs (worse; kept default-off as MINISGL_CUSTOM_AR).
NVLS nondeterminism is the source of the observed ±8 tok/s instance band —
a deterministic <=8 µs multimem kernel is both faster and steadier.

Reproduce: mini-sglang `a26fd6f` on verda B300 x8; comm plugin interface at
`python/minisgl/distributed/impl.py`.
