# TileRT kernel tasks

CUDA optimization tasks for the **fused kernels of TileRT** (the closed-source,
B200-only, batch=1/TP=8 low-latency DeepSeek-V3.2 inference engine). Each task
gives: the problem definition, exact shapes/dtypes, a **correct PyTorch baseline**
(from TileRT's own `golden_forward` reference), and the **measured TileRT latency
+ HBM bandwidth** (from calling the real `libtilert_dsv32.so` op, profiled with
ncu on B200). The KDA goal per task: write a CUDA kernel that **matches TileRT's
measured latency** on the decode shapes.

Reverse-engineering background (read for context): `../TileRT_讨论材料.md`
(esp. §3 kernel taxonomy, §4 persistent-grid/warp-spec, §13 fusion, §16 decode
profile, §18 weight layouts, §22 per-kernel shapes).

TileRT design facts to exploit (from the blog + SASS):
- **Persistent grid, occupancy=1**: 148 CTAs (=B200 SM count) × 384 threads
  (256 Consumer + 128 Prefetcher), registers pushed to ~168 to force 1 CTA/SM.
- **Warp specialization + TMA double-buffer**: prefetcher warps stream weights
  GMEM→SMEM via TMA (`cp.async.bulk`/`UBLKCP`), consumer warps run tensor-core
  MMA; mbarrier (`ARRIVE.TRANS64`) handshake. Weights read **once**; intermediate
  activations stay on-chip (don't round-trip GMEM).
- Dominant MMA path is **warpgroup HMMA (BF16)**, not tcgen05 (tcgen05 only in the
  DSA sparse indexer). FP8/FP4 use dedicated MMA paths.

Tasks live under `kernels/<arch>_tilert_<family>/`, mirroring `../diffusion/kernels/`.

> Status: WIP. Baselines (golden_forward) are correctness-validated per-op vs the
> real TileRT op. The first task (head_proj_gemm) is complete as the template;
> remaining kernels' isolated ncu latency/bandwidth are being measured on B200.
