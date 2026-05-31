# Interface: b200_diffusion_qknorm_rope__multi_shape

- Kernel slug: `b200_diffusion_qknorm_rope__multi_shape`
- Op type: `qknorm_rope_inplace`
- Target GPU: NVIDIA B200 (SM100)
- Wrapped SGLang entry point: `sglang.jit_kernel.diffusion.qknorm_rope:fused_inplace_qknorm_rope`

## Recovered baseline contract (AC-1)

Recovered on `ion-b200` inside `sglang_bbuf` from the live SGLang checkout:
- SGLang version: `0.5.12.dev472+g3f7e538b2`, repo HEAD `0b65588c1` (`/home/sglang-omni/bbuf/repos/sglang`).
- Source: `python/sglang/jit_kernel/diffusion/qknorm_rope.py` (Python wrapper) + `python/sglang/jit_kernel/csrc/diffusion/qknorm_rope.cuh` (CUDA kernel `QKNormRopeKernel<...>::run` / `fused_qknorm_rope_warp`).
- Oracle reference test: `python/sglang/jit_kernel/tests/diffusion/test_qknorm_rope.py`.
- Benchmark reference: `python/sglang/jit_kernel/benchmark/diffusion/bench_qknorm_rope.py`.

### Public signature (verified by import)

```python
@register_custom_op(mutates_args=["q", "k"])
def fused_inplace_qknorm_rope(
    q: torch.Tensor,            # [num_tokens, num_qo_heads, head_dim], bf16/fp16, last dim contiguous
    k: torch.Tensor,            # [num_tokens, num_kv_heads, head_dim], same dtype/stride contract
    q_weight: torch.Tensor,     # [head_dim], same dtype as q
    k_weight: torch.Tensor,     # [head_dim], same dtype as q
    cos_sin_cache: torch.Tensor,# [max_position, rope_dim], float32, contiguous last dim
    positions: torch.Tensor,    # [num_tokens], int32 or int64
    *,
    is_neox: bool,
    eps: float = 1e-6,
    head_dim: int = 0,          # defaults to q.size(-1)
    rope_dim: int = 0,          # defaults to cos_sin_cache.size(-1)
) -> None
```

- **Return / mutation contract**: returns `None`; mutates `q` and `k` **in place** (registered with `mutates_args=["q","k"]`). Correctness must be checked on the post-call `q`/`k` tensors, never on a return value.
- `head_dim` defaults to `q.size(-1)`; `rope_dim` defaults to `cos_sin_cache.size(-1)`.
- Internally dispatches to a JIT kernel templated on `(head_dim, rope_dim, is_neox, is_arch_support_pdl(), dtype)`, then calls `module.qknorm_rope(q, k, q_weight, k_weight, cos_sin_cache, positions, eps)`.

### `cos_sin_cache` layout (verified)

Built by `create_cos_sin_cache(rope_dim, max_position)`:
- `inv_freq = 1 / base**(arange(0, rope_dim, 2)/rope_dim)` → length `rope_dim/2` (`base=10000`).
- `freqs = outer(arange(max_position), inv_freq)` → `[max_position, rope_dim/2]`.
- `cos_sin_cache = cat((freqs.cos(), freqs.sin()), dim=-1)` → `[max_position, rope_dim]`.
- Row `pos` = `[cos(0..rope_dim/2) | sin(0..rope_dim/2)]`. The kernel reads `cos_ptr = cache + pos*rope_dim`, `sin_ptr = cos_ptr + rope_dim/2`.

### Algorithm (from `fused_qknorm_rope_warp`)

- 256 threads/block = 8 warps; **one warp processes one `(token, head)` head-vector**; grid-stride loop over `num_works = (num_qo_heads + num_kv_heads) * num_tokens`. `k_ptr` is pre-offset by `-num_qo_heads*head_stride` so a single `head_id` in `[0, num_qo+num_kv)` indexes q then k. PDL-enabled when arch supports it.
- `kElemsPerThread = head_dim/32`; each lane loads `AlignedVector<bf16x2, kElemsPerThread/2>` = `head_dim/32 * 2` bytes (head_dim=128 → 8 bytes / 64-bit per lane).
- **RMSNorm (fp32)**: per-lane sum of squares → `warp::reduce_sum` → `norm_factor = rsqrt(sum_sq/head_dim + eps)`; then `elem *= norm_factor * weight` (weight upcast to fp32). Normalizes over the **full head_dim**; `eps` is inside the sqrt.
- **RoPE**, only on lanes `< rope_dim/elems_per_thread`:
  - `is_neox=False` (interleaved / GPT-J): for element pairs `(2j, 2j+1)` in a lane, `half_idx = (lane*elems_per_thread + 2j)/2`; `x' = x*cos[half_idx] - y*sin[half_idx]`, `y' = y*cos[half_idx] + x*sin[half_idx]`.
  - `is_neox=True` (half-split): `__shfl_xor` by `rotary_lanes/2`, negate lower half, `elem' = elem*cos + swapped*sin`.
- Stores back as the input dtype (bf16/fp16).

### Support gate (`can_use_fused_inplace_qknorm_rope`)

Returns False (→ caller must fall back to baseline) unless ALL hold, then tries a JIT load:
- `head_dim in {64, 128, 256}`.
- `0 < rope_dim <= head_dim`.
- `rope_dim % (head_dim/32) == 0` (per-lane width divides rope_dim).
- if `is_neox`: `rotary_lanes = rope_dim/(head_dim/32)` must be `>= 2` and a power of two.
- `dtype in {bf16, fp16}` (kernel `static_assert`).

For the production signature (`head_dim=128, rope_dim=128, is_neox=False, bf16`): `elems_per_thread=4`, `128 % 4 == 0`, supported.

## Oracle (correctness reference, from the SGLang test)

`split_qknorm_rope` = `fused_inplace_qknorm(q,k,q_weight,k_weight[,eps])` then
`flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(positions=positions.long(), query=q.view(N,-1), key=k.view(N,-1), head_size=head_dim, cos_sin_cache=cos_sin_cache, is_neox=is_neox)`.
- `sglang.jit_kernel.norm.fused_inplace_qknorm(q, k, q_weight, k_weight, eps=1e-6, *, head_dim=0)` — RMSNorm over head_dim, **eps is the 5th positional arg** (must be passed for the `eps=1e-5` zimage cases).
- Tolerance ceiling: `ATOL=8e-2, RTOL=1e-2` (the SGLang test compares its own fused path vs this split oracle at this tolerance; the split path differs by ~1 bf16 rounding step).

## Optimized candidate interface

`src/register.py` exposes `optimized_wrapper(*args, **kwargs)` preserving the signature above, a `register()` dict, and (for promotion) `EXPORTS = {"fused_inplace_qknorm_rope": <wrapper>}`.

- The candidate is a workspace-owned native CUDA kernel (clean-room, built via `torch.utils.cpp_extension`), not a dependency on SGLang-internal headers.
- Fast path covers the production signature; everything else falls back to the SGLang baseline (`fused_inplace_qknorm_rope`), with the baseline reference bound at import time so the fallback never recurses after `kda_kernels.install()` swaps the symbol.

### Final result (promoted)

- **Final wrapper signature** (matches the recovered baseline exactly):
  `fused_inplace_qknorm_rope(q, k, q_weight, k_weight, cos_sin_cache, positions, *, is_neox, eps=1e-6, head_dim=0, rope_dim=0) -> None` (in-place; `src/wrapper.py`).
- **Promoted kernel**: candidate v3 = 2-heads-per-warp, 128-bit (`float4`) vectorized
  fused RMSNorm(fp32)+RoPE, half-warp RMS reduction, coalesced `float2`/`float4`
  cos/sin (`src/csrc/qknorm_rope_kernel.cu`, sha1 `021eb444`).
- **Dispatch table**: single native CUDA path for the production signature (bf16,
  contiguous, head_dim=128, rope_dim=128, is_neox=False, equal Q/K heads,
  int32/int64 positions); all other tuples → SGLang baseline fallback. See
  `docs/dispatch.md`.
- **Fallback cases** (verified routing to baseline): head_dim 64/256, rope_dim <
  head_dim, is_neox=True, non-bf16, non-contiguous, foreign-device, unequal Q/K
  heads. int32 positions at the production signature run natively.
- **Trusted precondition** (matches the SGLang baseline, not re-validated on the
  hot path): each `positions[i]` must be a valid row index into `cos_sin_cache`
  (`< cos_sin_cache.size(0)`). The baseline also indexes `cos_sin_cache[pos]`
  without a bounds check; adding a per-call device max-reduction would regress
  this latency-bound kernel below the baseline. Cheaply-detectable cache
  malformations (rank/dtype/last-dim/device/contiguity) are rejected by the gate
  and fall back; the `test_dispatch_gate` negative test covers them.
- **Tolerance methodology**: candidate vs SGLang split oracle
  (`fused_inplace_qknorm`(eps) + flashinfer RoPE) under hard ceiling `ATOL=8e-2,
  RTOL=1e-2`, plus a dynamic fp32-anchored bound (candidate-vs-fp32 error ≤
  `max(2e-3, 4× oracle bf16-vs-fp32 noise)`), with NaN/Inf checks. 21/21 pass.
- **Benchmark command**: `CUDA_VISIBLE_DEVICES=<idle> KDA_RUN_CORRECTNESS=1
  python benchmark.py` (CUDA-event timing of ONE kernel invocation per sample,
  with a pristine in-place Q/K reset + sync before each sample's `start.record()`,
  warmed modules; GPU util/mem recorded before and after settling to idle).
  Per-call latency = median over the single-invocation samples (`elapsed_ms*1000`
  µs); geomean = `exp(mean(ln(per-shape baseline_median/candidate_median)))`.
- **Result** (per-call benchmark, pristine Q/K reset per timed sample): geomean
  speedup over baseline **all 1.111×, large 1.133×, tiny 1.091×** on NVIDIA B200;
  correct on all 10 production shapes. Active bound: latency (cold ~36–45% of HBM
  peak); see `profile/cand_qwen4096_v3/REPORT.md`.
- **Source lineage**: algorithm ported clean-room from SGLang
  `fused_qknorm_rope_warp` (`csrc/diffusion/qknorm_rope.cuh` @ 0b65588c1) via
  torch cpp_extension; NCU rule engine drove the v2 coalescing fix; cold-cache
  roofline drove the v3 MLP (128-bit/2-head) change. Recorded in `solutions.jsonl`.
- **Promotion**: `EXPORTS={"fused_inplace_qknorm_rope": ...}` →
  `scripts/export_kda_kernels/export.py b200_diffusion_qknorm_rope__multi_shape`;
  install()/uninstall()/status() + post-install recursion-safe fallback verified
  on B200 (`verify_promotion.py`).

## Bench / roofline notes
- B200: 148 SMs, ~8 TB/s HBM3e. Dominant traffic per `(token, head)` = read+write of one head-vector = `2 * head_dim * sizeof(dtype)` (= 1024 B for bf16 head_dim=128 across q-read+q-write... per work it is 512 B; total `num_works * 512 B`). `cos_sin_cache` rows (`rope_dim*4` B/token) are L2-shared across heads; weights broadcast.
- Tiny-token shapes (19/32/47) are launch/latency-bound, not bandwidth-bound; report buckets separately.
