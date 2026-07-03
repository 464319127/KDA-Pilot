# KDA-Pilot mega_kernels — GLM-5.2 bs=1 decode (B300)

Fixed-shape kernel tasks distilled from the **mini-sglang GLM-5.2 bs=1
throughput project** (repo `BBuf/mini-sglang`, branch `glm5.2-support`,
baseline commit `a26fd6f`). Unlike `llm/` (shapes harvested from live serving
at mixed concurrency), every task here has a **single fixed decode
configuration**, so kernels can be specialized aggressively (constant shapes,
persistent kernels, full-占卡 tuning).

## Deployment being optimized

- Model `zai-org/GLM-5.2-FP8` (751B MoE, 78 layers + 1 MTP layer), TP=8 on
  8x NVIDIA B300 SXM6 (sm_103), NVLink NVLS.
- MTP speculative decode, cookbook-aligned (sglang GLM-5.2 FP8 cookbook uses
  `--speculative-algorithm EAGLE --speculative-num-steps 5
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`; mini-sglang
  equivalent `MINISGL_SPEC_STEPS=5..7`, chain topk 1, verify batch = steps+1).
- Reference config for all shapes below: **k=7 drafts, verify batch M=8,
  chain steps M=1**, page_size 64, greedy.
- Baseline end-to-end: **292.6 tok/s decode** (accept 5.16, round ≈ 17.6 ms).
  TileRT (closed reference) ≈ 500 tok/s on the same hardware class.

## Fixed shapes (memorize; every task pins a subset)

| symbol | value | meaning |
|---|---|---|
| H | 6144 | hidden size |
| E | 256 | routed experts per rank (TP slices the FFN dim) |
| I | 256 | routed expert intermediate per rank |
| topk | 8 | experts per token (noaux_tc sigmoid gate, routed_scaling folded in weights) |
| M | 1..8 | decode rows: verify=8, MTP extend=1..8, chain=1 |
| kv_lora / rope | 512 / 64 | MLA compressed KV + rope dims (combined pool, page 64) |
| q heads (rank) | 16 | q heads per rank after TP=8 |

## Round anatomy at k=7 (per ~17.6 ms round, measured/estimated)

| component | per-layer @M=8 | per-round | share |
|---|---|---|---|
| MoE routed (triton gemm1+gemm2, tuned) | ~47 µs | ~3.5 ms | 20% |
| MoE glue (quant/align/sum, post-`tiny_align`) | ~8 µs | ~0.6 ms | 3% |
| dense GEMMs (qkv_a/q_b/o/shared/lm_head, nvjet) | ~30 µs | ~2.4 ms | 14% |
| MLA decode attention (trtllm-gen sm_100f cubin) | ~17 µs | ~1.4 ms | 8% |
| all-reduce (NCCL NVLS, 2/layer, bf16 [8,6144]) | ~24 µs | ~1.9 ms | 11% |
| MTP extend + 6 chained draft steps (fused chain graph) | — | ~2.0 ms | 11% |
| norms/rope/store/router/sampler/gaps | — | rest | ~33% |

## Rules

Follow `../llm/docs/llm_kernel_optimization_rules.md` (native CUDA only, no
DSL candidates) and `../llm/docs/llm_correctness_contract.md`, with these
subtree-specific overrides:

1. **Target GPU is B300 (sm_103)**, not B200. tcgen05/TMEM, cp.async, PDL,
   NVLS multimem are all available. Cubins that only ship sm_100 AOT images
   (flashinfer TGV, cutlass_mla) do NOT run — check before depending on one.
2. **Baselines come from mini-sglang**, not upstream sglang, when the task
   says so: copy the exact file from `BBuf/mini-sglang@glm5.2-support` into
   `baseline/` and record the commit in `docs/baseline_source.md`.
3. **Accept-preservation contract (critical, hard-won).** The deep MTP chain
   makes end-to-end throughput depend on the *numerics style* of the main
   model, not just correctness: the accept distribution is bimodal and ~40%
   of rounds are full-accepts that die under per-layer numerics shifts as
   small as ~1e-3 elementwise (moving a routed-weight multiply across a bf16
   round-trip collapsed accept 5.16→3.3 while GSM8K stayed 93-95%). Therefore
   every task declares one of two deployment tiers in its `prompt.md`:
   - **tier A (bitwise)**: candidate must be bitwise-identical to the
     incumbent for main-model use (e.g. glue/reordering-free kernels), or
   - **tier B (accept-gated)**: candidate changes numerics; it may only ship
     after an in-server A/B shows accept (and the full-accept histogram bin)
     is preserved, or it ships to the MTP draft layer only (draft-side
     numerics changes are safe — a more accurate draft kernel *raised*
     accept 5.0→5.2).
   Kernel-level speedups are still the KDA success metric; the tier only
   dictates how the win is deployed.
4. **Benchmark in a CUDA graph** (see `../llm/docs/standalone_llm_benchmark.md`
   plus the local harnesses copied into `bench/`): eager timing overestimates
   these tiny kernels ~8x. Lock clocks, warm up, use graph replay medians.
5. Weights for oracle runs: random fp8/bf16 tensors of the pinned shapes are
   sufficient; real-checkpoint scales vary ~1e-4..1e-1 so sample block scales
   log-uniformly in that range, not uniform-narrow.

## Task index

See `_INDEX_glm52_bs1.md` for the ranked list with evidence.
