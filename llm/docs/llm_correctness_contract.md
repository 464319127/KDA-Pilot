# LLM Kernel Correctness Contract

How to build each task's `bench/correctness.py`. Tests run against the copied
local `baseline/` and an independent PyTorch/math oracle; they must not import,
patch, or monkey-patch SGLang at runtime.

The concrete regression grid for a task = **every deduplicated captured variant**
in its `prompt.md` / `docs/evidence.json` (production rows) **plus** the edge rows
listed for its category below. A candidate must pass the full grid before any
benchmark counts.

## General Rules (all kernels)

- For every output, check: shape, dtype, device, **stride/contiguity match vs the
  baseline's output**, NaN/Inf, and value tolerance.
- **Poison output buffers** (fill with NaN / a sentinel) before each run so
  stale-output, partial-write, and skipped-kernel bugs are visible.
- Compare candidate vs an independent oracle when practical; at minimum compare
  candidate vs the copied baseline plus targeted oracle rows.
- Respect the captured scalar kwargs (e.g. `eps`, `is_neox`, `rope_dim`, group
  size, `is_rms_norm`). Do not hardcode one value.
- Cover the captured **strides** (see Non-contiguous inputs) and any **in-place**
  semantics (see In-place kernels), not just a clean contiguous/out-of-place case.

## Default Tolerances By dtype

Use these unless a task records a stricter task-local tolerance in
`docs/benchmark_method.md`:

| dtype | atol | rtol |
|---|---|---|
| fp32 | 1e-5 | 1e-5 |
| fp16 | 3e-3 | 3e-3 |
| bf16 | 7e-2 | 2e-2 |
| fp8 (e4m3/e5m2) dequantized | compare in the **dequant domain**: scales bit-exact-or-1-ulp; dequant(q)·scale within bf16 tolerance of the reference | — |

Integer / index / boolean / tree-structure outputs are **exact-match**, not
atol/rtol (see Top-k, Speculative tree, KV-cache store).

## Per-category contracts

### Norm — `sgl_kernel_rmsnorm`, `sgl_kernel_fused_add_rmsnorm`, `srt_layers_layernorm_layernorm`
- Oracle: reference RMSNorm / LayerNorm in fp32 accumulation, cast to output dtype.
- `fused_add_rmsnorm` writes the residual back in place: verify **both** the
  normalized output **and** the updated residual, and test input/residual
  aliasing exactly as captured.
- Edge rows: `rows ∈ {1, 2, 38, large captured M}` × `hidden ∈ {captured sizes, e.g. 512, 2048, 6144}`; dtype bf16 (+ fp32 oracle); `eps` as captured.

### RoPE — `jit_kernel_rope_apply_rope_inplace`
- Oracle: FlashInfer-style `apply_rope_with_cos_sin_cache`. Honor `is_neox`
  (neox/half-half vs interleaved/GPT-J adjacent-pair) — they are different
  rotations; test both if both are captured.
- In-place over q and k views; cos/sin cache is fp32; positions are int. Test the
  captured `rope_dim` (incl. partial-rotary `rope_dim < head_dim`).
- Edge rows: token counts `{1, 2, 38, captured}`; head layouts `{1, 32, 64}×64`;
  `is_neox ∈ {captured}`; bf16.

### Activation — `jit_kernel_activation_run_activation_inplace`
- Oracle: the exact activation captured (SwiGLU/SiLU·mul or GeLU). In-place: the
  output aliases part of the input (gate/up split) — verify the in-place layout.
- Edge rows: `rows ∈ {1, 2, captured}` × intermediate sizes captured; bf16.

### Quantization — `jit_kernel_per_token_group_quant_8bit_v2[_custom_op]`
- Two outputs: the **fp8 quantized tensor** and the **per-group scales**. Verify
  **both**. Oracle: reference per-token-group abs-max → scale → quantize, with the
  same group size, rounding mode, and clamp as the baseline.
- Check the group layout (group size, ragged last group if hidden % group ≠ 0)
  and the scale dtype/layout (block-scale).
- Acceptance: scales match to ≤1 ulp; `dequant(q)·scale` within bf16 tolerance of
  the reference pre-quant tensor; quantized ints exact where rounding is defined.

### FP8 GEMM — `srt_layers_quantization_fp8_kernel_deep_gemm_fp8_fp8_bf16_nt`, `sglang_quant_method_fp8_linear_method_apply`
- fp8 A/B + per-block/per-token scales → bf16 out. Oracle: dequantize A and B to
  fp32, matmul, apply scales, cast to bf16. Tolerance: bf16-level, but allow for
  fp8 accumulation order — use atol≈bf16 plus a small relative margin tied to K.
  Note: DeepGEMM is a strong baseline; correctness must match exactly even if a
  speed win is unlikely.
- Verify the scale layout (block size, NT orientation) matches the captured args.

### Attention — `srt_layers_attention_base_attn_backend_attention_backend_forward`
- Oracle: reference scaled-dot-product attention with the captured causal/mask,
  head/kv-head counts (GQA), head_dim, softmax scale, and KV layout. Tolerance:
  bf16. Verify the output AND any updated KV/state the baseline mutates.

### Top-k / sampling — `jit_kernel_grouped_topk_jit_grouped_topk_op`, `sgl_kernel_fast_topk_transform_fused`
- **Exact-match** on selected indices (and on values within tolerance for the
  gathered weights). Tie-breaking must match the baseline's rule exactly (equal
  scores → same chosen index); construct explicit tie rows. For grouped top-k,
  verify the group mask / group-then-topk semantics and renormalization.

### Speculative tree — `sgl_kernel_build_tree_kernel_efficient`, `sgl_kernel_verify_tree_greedy`
- Integer/structural outputs (tree parents, positions, accepted tokens, masks):
  **exact-match** vs baseline. Cover the captured tree width/depth and the
  accept/reject branches (full accept, partial accept, full reject).

### KV-cache store — `jit_kernel_fused_store_index_cache_fused_store_index_k_cache`
- Indexed scatter into cache slots. Verify by reading the cache back: the written
  slots equal the source (exact for integer/quantized, tolerance for float) AND
  untouched slots are unchanged (poison the cache first). Cover duplicate/edge
  indices as captured.

## Non-contiguous inputs

Most GLM-5.2 captures are `is_contiguous=False` (strided views from slicing
fused QKV / gate-up). The candidate must produce correct results on the captured
strides, or explicitly fall back to baseline for strided inputs and only claim
the contiguous speedup. Always include at least one strided row per kernel and
assert the candidate's output stride matches the baseline's.

## In-place kernels

`apply_rope_inplace`, `run_activation_inplace`, and similar mutate their inputs.
Tests must: snapshot inputs, run baseline and candidate on **separate copies**,
compare the mutated buffers, and verify aliasing (output view overlapping input)
behaves identically to the baseline. Poisoning still applies to any auxiliary
output.
