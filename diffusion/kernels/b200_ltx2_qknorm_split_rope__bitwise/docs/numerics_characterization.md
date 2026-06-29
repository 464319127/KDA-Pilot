# Numerics Characterization (bit-exact contract pinned to this stack)

Produced by `bench/characterize_numerics.py` on the B200. This pins the exact bf16
rounding behavior the candidate kernel must reproduce, and is the evidence for the
"characterize empirically before optimizing" gate.

## Pinned stack (bit-exactness is asserted against THIS stack)

- Host: `ion-b200` (`innomatrix-us-adc-smb200-0003`), container `sglang_bbuf_pr29315`,
  workspace `/tmp/ltx2_qknorm_task`.
- GPU: NVIDIA B200, compute capability (10, 0) / sm_100. GPU id 5 (idle: 0% util, 0 MiB).
- Python 3.12.3, torch 2.11.0+cu130, CUDA 13.0, tvm_ffi 0.1.9.
- If this stack changes, re-run the probe and re-validate (bf16 bit-exactness is
  stack-specific; see DEC-5).

## (A) RMSNorm(H) over the full hidden dim

Reference: `torch.nn.RMSNorm(H, eps=1e-6, dtype=bf16)` (bf16 weight). Probe data:
H in {256, 2048, 4096}, 8 seeds, 192 rows/seed, plus adversarial rows (all-equal,
single-outlier, boundary tiles) and boundary-targeted weights.

**Affine order — DECIDED:**
- `V1 = bf16(fp32(x) * rstd * fp32(weight))` — multiply the weight in fp32, with a
  SINGLE bf16 cast at the very end. (≈25 residual mismatches, all from the rstd
  1-ULP issue below — not from the affine order.)
- `V2 = bf16(bf16(fp32(x) * rstd) * fp32(weight))` — casting the normalized value to
  bf16 *before* the weight multiply is WRONG: ~2.41M mismatches. Ruled out.
- So: normalize and apply weight entirely in fp32, then one final `__float2bfloat16_rn`.

**rstd reduction — NOT bit-exact with a naive custom reduction:**
- Candidate `rstd = rsqrt(mean(x^2) + eps)` (torch reduction) differs from
  `torch.ops.aten._fused_rms_norm(x, [H], w, eps)[1]` by **max 1 ULP** (fp32).
- That 1-ULP rstd difference flips ~25 / 2.41M output bf16 values (where
  `x*rstd*weight` sits on a bf16 midpoint). `rsqrt` vs `sqrt`+reciprocal: rsqrt is
  closer (25 vs 49 mismatches); reduction method (mean vs sum/N) did not change the count.
- **Conclusion:** a fully-custom CUDA RMSNorm is NOT yet bit-exact — it would need to
  replicate PyTorch's exact mean-square reduction + rsqrt to 0 ULP. Until that is
  cracked, use **ATen RMSNorm** (PyTorch itself) for the norm stage; it is exact by
  construction. Matching PyTorch's rstd for a fully-fused kernel is a later-round
  optimization (DEC-4: staged fallback is acceptable).

## (B) split-RoPE (the addcmul_ rounding sequence)

Reference: `baseline/ltx2_split_rope.py:apply_split_rotary_emb_eager`. Probe cases:
(B,S,heads,head_dim) in {(1,2048,1,64),(2,257,32,64),(1,129,32,128),(1,512,32,128)},
6 seeds, boundary-value x/cos/sin in the production non-contiguous layout.

**DECIDED:** round `first*cos` (and `second*cos`) to bf16 first, then combine the sine
term in **fp32 with a single final bf16 cast**:
- `out_first  = bf16( fp32(bf16(first*cos))  -  fp32(sin)*fp32(second) )`
- `out_second = bf16( fp32(bf16(second*cos)) +  fp32(sin)*fp32(first)  )`
- Matching variants (0 mismatches vs eager): `A1_fp32_muladd` (separate fp32 mul then
  add) and `A2_fmaf` (single fp32 FMA) — both produce identical bf16 here, so the
  kernel may use either; we use explicit rounded intrinsics (`__fmul_rn`/`__fadd_rn`,
  or `__fmaf_rn`) and never let `-O3` decide.
- WRONG: `A3_bf16_product` — rounding `sin*second` to bf16 before the add (~1.44M
  mismatches). Do NOT round the sine product.
- Sensitivity guard: the variants disagree on every probe case (≥3 pairwise
  disagreements each), so a match is informative, not a coincidence.

## Implications for the kernel (strategy)

1. **Correctness floor (this round):** staged — ATen RMSNorm for q and k (bit-exact by
   construction), then ONE fused split-RoPE-pair CUDA kernel using the A1 sequence
   (bf16(x*cos) then fp32 combine, single cast; index cos/sin via real strides). This
   passes the full correctness suite and already removes the eager RoPE's multiple ATen
   launches.
2. **Optimization (later round):** to fully fuse RMSNorm into the kernel for the larger
   win, first crack the rstd 0-ULP match against `aten._fused_rms_norm`; if it cannot be
   matched after bounded attempts, keep ATen RMSNorm staged (DEC-4).
3. **B200 codegen:** use `__bfloat162float` / `__float2bfloat16_rn`, explicit `__fmul_rn`/
   `__fadd_rn`; never write `a*b+c` (could be contracted); preserve `-0.0` in `-sin` by
   negating before the float convert.
