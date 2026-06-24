**DESIGN_RANKED**

1. **M=1 vectorized FP8 GEMV, one warp per output column `n`**
   Highest payoff. This attacks the actual waste: baseline computes a 64-row tile for one row.

   Mapping:
   - Grid: `blockIdx.x` covers a tile of output columns `n`.
   - Block: 128 or 256 threads, preferably 4 or 8 warps.
   - Normal large-`N` mode: each warp owns one `n`; one CTA computes 4 or 8 output columns.
   - Addressing: `Bphys[n, k] = B + n * K + k`, so each warp reads one contiguous K-row of B.
   - K chunk: each lane loads `uint4` / 16 FP8 values at `k = ktile + lane * 16`. One warp therefore loads 512 contiguous B bytes per K step.
   - A reuse: preload the current A row, or K-tile of A, into shared memory once per CTA. For `K <= 8192`, full-row shared preload is feasible: 1 to 8 KB for `M=1`. All warps reuse it.
   - Reduction: per-lane FP32 partial accumulation over its 16-byte chunks, then warp shuffle reduction. Lane 0 applies `scale_a[0] * scale_b[n]`, converts to bf16, stores `out[n]`.
   - Expected use: `M == 1` only, plus maybe `M <= 4` after benchmarking.

2. **Swap-AB skinny CUTLASS/CuTe FP8 GEMM for `M >= ~8 or 16`**
   Compute logical transpose:
   - Original: `C[M,N] = A[M,K] * B[K,N]`
   - Use: `C^T[N,M] = Bphys[N,K] * A^T[K,M]`
   This makes the large dimension `N` the GEMM M dimension and the skinny dimension the GEMM N dimension. It avoids the worst 64-row M waste. Epilogue scatters to row-major `out[m * N + n]` and applies `scale_a[m] * scale_b[n]`.
   
   This is the best candidate for `M=23,32,57`. It matches the known SM100 direction from vLLM’s swap-AB FP8 work and M-dependent SM100 tuning in the local KernelWiki pages `sources/prs/vllm/PR-27284.md` and `sources/prs/vllm/PR-19566.md`.

3. **Multi-row GEMV for very small `M`, e.g. `2 <= M <= 8`**
   Same as #1, but one warp owns one `n` and accumulates `R` rows of A at once:
   - Load B once.
   - Keep `R` FP32 accumulators.
   - Load `A[m, k]` for `R` rows from shared/registers.
   - Store `R` outputs at `out[m * N + n]`.
   
   This reuses B across rows, but register pressure grows quickly. I would only specialize `R = 2, 4, 8`. Past that, swap-AB tensor-core GEMM is probably better.

4. **Low-N intra-CTA split-K GEMV variant**
   For `N=512/1024`, one warp per `n` gives too few CTAs. Use one CTA per output `n`, with 4 or 8 warps splitting K inside the CTA:
   - Warp `w` computes K slice `[w*K/W, (w+1)*K/W)`.
   - Reduce warp partials through shared memory.
   - One final bf16 store.
   
   This avoids global atomics and avoids a second kernel.

5. **Full custom tcgen05/TMEM skinny GEMM**
   Potentially fastest for `M=16..64`, but highest complexity. Not first. A swap-AB CUTLASS/CuTe kernel is the pragmatic path.

**BOTTLENECK**

For the top `M=1` GEMV, the intended bound is HBM bandwidth plus FP8 decode/FP32 FMA issue overhead.

For `M=1,K=1024,N=8192`:
- B traffic: `1024 * 8192 * 1 B = 8.39 MB`
- Output + scales: negligible, about 50 KB.
- A is tiny and should mostly hit L2/shared; even pessimistically it adds around 1 MB.

At 70 to 80% of 8 TB/s:
- `8.39 MB / 5.6 TB/s = ~1.50 us`
- `8.39 MB / 6.4 TB/s = ~1.31 us`

Realistic first target: **2 to 3 us**, because scalar FP8 unpack/convert and reduction overhead will keep it below ideal HBM roofline. Still far ahead of the measured 12.2 us baseline.

**SMALL_N_RISK**

`N=512/1024, M=1` has low CTA count if one CTA computes 8 columns. Use two modes:

- `N >= 2048`: CTA tile `N=4 or 8`, one warp per output column.
- `N <= 1024` and `K >= 1024`: CTA tile `N=1 or 2`, split K across 4 or 8 warps inside the CTA.

Avoid inter-block split-K initially. It needs FP32 partial workspace or atomics, adds traffic, complicates the ABI, and can lose determinism. Intra-CTA split-K gives more CTAs without extra global traffic.

**LARGE_M_NOTE**

For `M=23,32,57`, pure GEMV becomes unattractive because B is reread for each row group. A skinny tensor-core GEMM with swap-AB should win.

Dispatch recommendation:
- `M == 1`: custom GEMV.
- `2 <= M <= 8`: benchmark multi-row GEMV vs swap-AB; likely GEMV for `M <= 4`, maybe `M <= 8` when `N` is large.
- `M >= 16 and M <= 64`: swap-AB CUTLASS/CuTe FP8 GEMM.
- `M > 64` or unsupported layouts/tails: existing baseline.

The exact crossover should be measured, but I would expect **GEMV -> swap-AB around M=8 to 16**.

**VERIFY_RISKS**

- FP8 format must match the tensor’s actual `float8_e4m3` variant, especially finite-only / NaN behavior.
- B indexing must be `B[n * K + k]`, not `B[k * N + n]`.
- Accumulate in FP32; do not silently use half accumulation.
- Apply scales after the dot: `acc * scale_a[m] * scale_b[n]`.
- bf16 store must use correct round-to-nearest behavior expected by the oracle.
- `out[m * N + n]` is row-major; swap-AB epilogue must scatter correctly.
- K vector loads require 16-byte alignment and `K % 16 == 0`; listed K values are safe, but keep a scalar/vector tail fallback.
- N tile tail guards are still needed for non-multiple tile counts, though listed N values are friendly.
- Split-K with atomics should be avoided unless a deterministic FP32 partial reduction path is provided.

**RECO**

Implement first: **a native CUDA `M==1` vectorized GEMV kernel with two launch variants: large-N one-warp-per-column and small-N intra-CTA split-K.** It is the clearest win against the 64-row CUTLASS waste and should target ~2 to 3 us for `1x1024x8192`. In parallel or next, add a **swap-AB CUTLASS/CuTe path for `M=16..64`**, then benchmark the crossover for `M=2..8`.
