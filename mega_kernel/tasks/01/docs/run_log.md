# Run Log — mega_kernel task 01 (mnnvl_ar_jit_bs1)

Environment recovery and measurement provenance log. Every GPU measurement in this task records host, all-8-GPU state before/after, and exact commands. Local time is given in the devbox clock unless noted.

## Environment Preflight (2026-07-10, local session)

- Host: rx devbox `glm52-bs1-opt` (worker hostname `light-face-hides-fin-03-1`), reachable over ssh from the task workstation. No reacquire needed.
- GPUs: 8x NVIDIA B300 SXM6 AC, 275,040 MiB each. At preflight: utilization 0% on all 8; memory ~228 GiB used per GPU — the resident GLM-5.2-FP8 serving instance is loaded and idle (baseline config, no jit flag; restart wrapper visible in process table with the documented `launch_devbox.sh` command and env `SGLANG_BS1_BF16_DENSE=1 SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1 SGLANG_BS1_FP8_DEFER_FINALIZE=1`, logging to `server_full.log`; log tail: "The server is fired up and ready to roll!").
- CUDA toolkit: release 13.0, V13.0.88 (`nvcc --version`). torch 2.11.0+cu130, `torch.version.cuda = 13.0`, `torch.cuda.device_count() = 8`.
- Topology: `nvidia-smi topo -m` shows NV18 between every GPU pair (full mesh), NUMA split 0-3 / 4-7.
- Peer access: `torch.cuda.can_device_access_peer(i, j)` True for all 56 ordered pairs.
- NVLS/cuMulticast: `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 1` on all 8 devices (cuda-python driver query).
- flashinfer: 0.6.12 at `/usr/local/lib/python3.12/dist-packages/flashinfer`.
- Kernel source present: `/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh`, 1223 lines, sha256 `ab6560f28bf94d06ce5ee80bd8414674d9d82c72fba6c8a19454b9c6ea1297aa`.
- Remote task workdir created: `/scratch/kda_bs1/mega01` (sibling task dirs `mega02`, `k2*` present — untouched).
- Serving bench assets confirmed at `/scratch/glm52_blog_bench/`: `launch_devbox.sh`, `launch_dummy.sh`, `benchmark_glm52_bs1.py`, `profile_devbox.sh` (bounded ~0.9 s `/start_profile` window on port 30000 while one greedy streaming request runs), `profiles/full376/` (original capture, TP-0..7 traces), `patches_main/`.

## Mandatory Reading (completed before implementation)

- `llm/docs/standalone_llm_benchmark.md`, `llm/docs/llm_kernel_optimization_rules.md`, `llm/docs/llm_correctness_contract.md` — ingested; load-bearing clauses quoted in `docs/benchmark_method.md` together with task deviations D1-D4 (single-GPU wording vs world=8 collective).
- `llm/docs/standalone_llm_benchmark_template.py` — to be copied verbatim as `bench/benchmark.py`; adapter contract and timing-policy fields mirrored by the task harness.
- `external/KernelWiki/SKILL.md` — query via `scripts/query.py` / `get_page.py`; flashinfer MNNVL lineage pages: `sources/prs/flashinfer/PR-1321.md`, `PR-2118.md`, `PR-2130.md`.
- `external/ncu-report-skill/SKILL.md` — profile -> diagnose -> plan; per-run directory convention; sm_100 metric-name caveats; `-lineinfo` harness rule. Task-specific: NCU on this spin/flag kernel must use `--replay-mode application` (kernel replay deadlocks).
- `external/warp-specialization-report-skill/SKILL.md` — predict (feeds & speeds) -> stamp clock() -> reconcile loop; applies only if a candidate becomes warp-specialized; the stock oneshot kernel is not warp-specialized (uniform role per thread), so N/A unless a P1 candidate introduces producer/consumer warp roles.
- Prior art: `mega_kernel/common/prior_art/old02_ar_norm_RESULTS.md` (unicast push dead: 36.6 us, ~25 us world-independent protocol cost; NVLS multimem is the only fast route), `mega_kernel/LEARNINGS.md`, `mega_kernel/integration/README.md` (hook point + 4-rung validation ladder).

## K/R/W Recovery

- K (kernel + callsite): `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel` — template `<uint8_t WorldSize, typename T, bool RMSNormFusion, typename PackedType = float4>`, `__launch_bounds__(1024)`, Lamport-buffer protocol (`LamportBufferLayout`, `LamportFlags` aligned(32), negative-zero sentinel via `isNegZero`, cooperative_groups), dispatched by `oneshotAllreduceFusionDispatch(AllReduceFusionParams const&)`. The header also contains a twoshot path (`twoshotAllreduceKernel` + `rmsNormLamport`) — NOT the target; bs=1 serving uses the oneshot fused path. Python assembly: `flashinfer/comm/trtllm_mnnvl_ar.py` (kernel wrapper) + `flashinfer/comm/mnnvl.py` (McastGPUBuffer / fabric + multicast workspace). sglang callsite: `python/sglang/srt/layers/flashinfer_comm_fusion.py` (serving checkout pinned at main `87992eeec` + `patches_main/main_port_full.diff`). Fused semantics (frozen): `out = rmsnorm(allreduce(x) + residual, weight, eps=1e-5)`, `residual_out = allreduce(x) + residual`, bf16, H=6144, weight[6144], world=8.
- R (oracle + baseline): flashinfer 0.6.12 original driven through the same 8-GPU harness = bit-exact reference; composed fp32 oracle (fp32 cross-rank sum + replicated residual, fp32-accumulation rmsnorm, cast bf16, rel < 1e-3) as documented fallback/sanity per plan AC-4.1.
- W (workloads + method): frozen rows T=6 (73.7 KB payload) and T=1 (12.3 KB), H=6144, bf16, world=8, eps=1e-5; benchmark mode per config.toml — 8x per-device 50-round CUDA graphs, concurrent replay, wall/round (pattern `prior/ar_bench.py`); 1000-replay bitwise stability as race detector; NCU application-replay only.

## SHAPES.md Re-verification (M0 Step 3) — COMPLETE

- Existing capture `profiles/full376/1783659367.6875372-TP-0.trace.json.gz` via `analyze_trace.py`: `oneshotAllreduceFusionKernel` count 9440, mean 8.3 us, total 78.55 ms; 9440 / 60 iters = 157.3 calls/iter — matches SHAPES.md exactly.
- Fresh bounded capture (documented protocol): `./profile_devbox.sh task01_shapes 0.9 1200` — one greedy streaming request (287 chunks), 0.9 s `/start_profile` window; trace `profiles/task01_shapes/1783671900.425971-TP-0.trace.json.gz`: count 13440, mean 8.5 us, total 113.57 ms. Consistent with the frozen table within sampling noise.
- Static verification: `/scratch/models/glm52-fp8/config.json` -> `rms_norm_eps = 1e-05`, `hidden_size = 6144`; callsite `flashinfer_allreduce_residual_rmsnorm` (custom op, mutates input/residual/weight; returns norm_out + residual_out; mnnvl backend auto-selected on SM103). eps discrepancy resolved: frozen 1e-5 confirmed authoritative (`prior/ar_bench.py`'s 1e-6 is stale prototype config).
- Reference-latency discrepancy resolved: in-graph serving mean is 8.3-8.5 us (8.7 us in old02 was the earlier round's number); the gate-governing immutable baseline comes from this task's harness measurement (task3).
- GPU state after profiling: all 8 GPUs back to 0% util, memory unchanged (resident serving healthy, log tail "fired up and ready to roll").
- SHAPES.md updated with a dated re-verification section; no shape/dtype/eps/world changes.

## Harness Bring-up (M1, 2026-07-10)

- Task workspace synced to `glm52-bs1-opt:/scratch/kda_bs1/mega01/task01` (tar over ssh; box lacks rsync).
- Single-process NVLS workspace (`bench/sp_nvls_workspace.py`) validated end-to-end via `bench/smoke_fi.py`: multicast create/AddDevice(8)/BindMem(8)/map OK; flashinfer ORIGINAL oneshot fused kernel ran eagerly on all 8 ranks against this workspace; Lamport flags rotate exactly per spec (`[0,2,..]` -> `[1,0,..,589824,..]` -> `[2,1,..]`; 589824 = 6*6144*8*2 B, the kernel-computed clear size); cross-rank outputs bitwise identical; round 2 on rotated buffers clean. Multicast granularity: RECOMMENDED = 512 MB (fabric page) — switched to MINIMUM -> 6,291,456 B total (3 x 2 MB Lamport buffers) per device.
- Measured bf16-vs-fp32-oracle floor: flashinfer ORIGINAL max-rel(out) = 3.904e-3 = 2^-8 = 1 bf16 ulp (T=6; 4.741e-3 after 3 rounds, T=1 3.942e-3) — the source prompt's fallback wording "fp32 oracle rel<1e-3" is unsatisfiable for any bf16 kernel under max-rel; oracle gate operates at the correctness contract's bf16 tolerance (atol 7e-2 / rtol 2e-2). Logged in goal tracker Plan Evolution; primary bitwise A/B gate unaffected.
- Comparator self-tests PASS (bit-flip caught; wrong-eps caught in the eps-dominated regime; poison caught; no false positives).
- Noise floor (`--mode noise`, fi vs fi, 7 trials, reps=10, rounds=50, pdl=0, resident serving request-idle at 0% util): T=6 A=6.713us B=6.801us pseudo-speedup=0.9871 (spread 1.29%); T=1 A=6.532us B=6.579us pseudo-speedup=0.9928 (spread 0.72%). Isolated back-to-back wall/round reads lower than the 8.3us in-serving figure as expected (3-buffer cross-round pipelining, no neighbor kernels); both impls are compared under the identical convention. Decision per D1: request-idle coexistence acceptable at this noise level; will stop serving and remeasure if any gate decision lands within the noise margin.
- GPU state before/after these runs: all 8 GPUs 0% util, memory at the resident-serving baseline (recorded in bench/results.jsonl records).

## Independent Reviews (analyze tasks, Codex gpt-5.5 xhigh, 2026-07-10)

- Harness/method review: AGREED on A/B symmetry, workspace protocol fidelity, oracle-tolerance reasoning, timing shape, noise-floor approach, D1-D4 justification. REQUIRED_CHANGES (all implemented before any further gate evidence): (1) stability detector hardened — 3 rotating input banks + in-graph output poisoning + per-bank references (constant-input/reused-output blind spots fixed), plus plain max-overlap variant; the first-cut "25k rounds stable" claim was downgraded and the gate re-run; (2) bench mode now refuses to time unless the two impls are bit-exact right then (recorded per row); (3) provenance extended (raw samples, headline geomean, GPU names/totals, nvcc + tvm-ffi versions, source hashes of baseline header + port files, candidate build flags); (4) stale "rel<1e-3" wording purged from docstrings; (5) result artifacts synced back into the repo tree. Full review: .humanize/skill/2026-07-10_16-51-37-*/output.md.
- Port-diff review: VERIFIED the kernel header differs from upstream ONLY in the include block and the binding only in include+comment; FFI drive parity between fi_original.py and jit_port.py confirmed; compat helpers behavior-equivalent on success paths; SGL_CUDA_ARCH define inert for this TU. DIVERGENCES: error-path message cosmetics only. REQUIRED_CHANGES: none. Notes for later shapes: twoshot/large-H paths inherited but untested here (out of task scope; serving routes only the oneshot decode regime to the port). Review: .humanize/skill output alongside.

## Serving Gate (P0.c) — COMPLETE (final verdict in docs/results.md serving-gate table)

- Baseline sanity (resident long-warm server, pre-patch): overall mean_decode_output_tok_s = 385.93; 40 records captured (results_task01_base).
- Patch applied to /sgl-workspace/sglang via exact-string patcher (pre-apply repo diff fingerprint aa8500cd... recorded; flashinfer_comm_fusion.py 77b14b94 -> 1ffe7044; new files: jit_kernel/mnnvl_ar_fused.py + csrc/mnnvl_ar_fused/). jit module prebuilt for sm_103a (override_jit_cuda_arch(10,3,"a")).
- Flag-OFF phase (patched code, env unset): server restarted healthy; sanity overall = 376.53 tok/s (>= 376); text records 40/40 identical to base (tokens, chars, prefix) -> OFF-inertness confirmed. Note: fresh-restart throughput sits ~2.4% below the long-warm base server; restart variance matters when reading the 376/378 thresholds.
- Flag-ON phase (route ON, jit module): 372.94 tok/s (< 376) and text records only 4/40 identical to base -> GATE FAILED; greedy outputs diverge under the routed path. Serving restored immediately afterward (restore phase: 376.70 tok/s, 40/40 identical to base — resident serving proven back at baseline behavior).

## Flag-ON Divergence Diagnosis (2026-07-10) — RESOLVED (root cause: baseline binary fast-math flags; fix verified end-to-end)

Facts established, in order:
1. OFF and restore phases are each 40/40 text-identical to base across full server restarts -> serving is restart-deterministic on the stock path; the ON divergence is caused by the routed path, not restart noise.
2. Serving callsite audit (layernorm.py `_forward_with_allreduce_fusion`): passes eps=variance_epsilon, never passes use_oneshot/fp32_acc/trigger_completion_at_end -> defaults None/False/False. flashinfer's unified `allreduce_fusion` mnnvl branch (read from installed 0.6.12) forwards to the same legacy fused call I mirror, silently dropping use_oneshot (AUTO -> oneshot for decode sizes). Route argument assembly is semantically identical to stock.
3. Harness experiments, all bf16 BIT-EXACT vs flashinfer original on all ranks/rows: (a) my build at pdl=1; (b) the SERVING-BUILT .so (sglang load_jit, arch 10.3a) at pdl=0; (c) the serving-built .so at pdl=1. Modules and pdl exonerated in isolation.
4. Hardened stability (3 rotating input banks, in-graph NaN poisoning, per-bank refs): 50,000 distinguishable-data rounds per row at pdl=1 + 1000 plain replays -> zero mismatches (ran under partial GPU contention, an added stressor). Kernel race-clean in the harness.
5. Serving bisect1 (route mechanics ON + STOCK module via SGLANG_JIT_MNNVL_AR_USE_STOCK_MODULE=1): 377.04 tok/s, 40/40 text-identical to base -> route mechanics (env checks, early return, custom-op body change, output allocs) exonerated IN SERVING.
6. Net: divergence requires MY .so EXECUTING inside the serving process; every isolated equivalence test passes. Remaining hypothesis space: call-class-specific behavior (attn-TP vs MoE-TP workspace managers), in-situ race visible only amid the full serving graph, or a module-cohabitation effect (both .so files loaded in one process: weak-symbol/registry interactions are value-neutral for bit-identical code, but scheduling metadata could differ).
7. Bisect2 (SCOPE=attn, my module): 4/40 vs base on BOTH sanity runs, run1-vs-run2 40/40 identical -> the divergence is DETERMINISTIC (not a race) and reproduced by the attn-AR class alone.
8. Discovery: the "flashinfer original" at runtime is the AOT wheel binary `flashinfer_jit_cache 0.6.12+cu130` (`get_library_path()` prefers it; there is NO locally-JIT-built module) — the harness "fi" side loads the same AOT binary, so all harness A/B comparisons were against the true production kernel.
9. Value-zoo A/B (bench/value_zoo_ab.py — inputs laced with bf16 subnormals, +-0.0, extremes; classes randn never produces): fi vs jit DIVERGE deterministically; every mismatch is an output zero-sign flip (0x0000 <-> 0x8000), only on `out` (never `residual_out`) -> the divergence lives in the gamma-multiply epilogue.
10. Kernel symbol signatures identical in both binaries (weightBias parameter present in both) -> same source revision; codegen-level difference.
11. SASS inspection: the AOT wheel is a fast-math build — 96,240 `.FTZ` instruction modifiers (FADD.FTZ throughout the reductions) and MUFU.RCP approximate division; my TVM-FFI build was IEEE (558 FTZ). Mechanism: for bf16-SUBNORMAL gamma values, AOT `FADD.FTZ(weightBias=+0, gamma_subnormal)` flushes to +-0 before the multiply while the IEEE build keeps the exact tiny value whose product casts to the opposite zero sign -> output zero-sign flips; randn inputs contain no subnormals, hence harness blindness. The approximate division (fullSum/tokenDim) is a second, rarer divergence channel.

ROOT CAUSE: build-flag asymmetry between the deployed baseline binary (fast-math AOT wheel) and the port's IEEE build — the verbatim SOURCE is identical; the float CODEGEN was not. FIX: compile the port with `--use_fast_math` (baseline-matching, not one-sided: the baseline binary carries the same float semantics), keep the source verbatim. Revalidation chain: value zoo -> randn correctness -> parity -> hardened stability -> serving flag-ON gate.

## Devbox Release + Reacquire (2026-07-10, during the promote pass)

- Mid-promote, rx began returning `devbox not running (status=releasing)` — the box was auto-released. Handled per the runbook (not a termination condition): `rxp devbox acquire --gpu B300 --count 8 --image lmsysorg/sglang:latest --name glm52-bs1-opt` + `extend` + `ssh-config`, then the three glm52-bs1-opt Host entries in `~/.ssh/rx_config` were proxychains-wrapped exactly like the `bbuf-gb300-8x` entries. Landed on the SAME worker (`light-face-hides-fin-03-1`) with a fresh pod: /scratch wiped, /personal intact (production patch + full `glm52_blog_bench` backup + a kda_bs1 snapshot), /sgl-workspace at the image default commit.
- Data continuity: all task code/docs and the documented gate numbers live in the repo worktree (synced back throughout). Lost with the pod and RE-COLLECTED on the rebuilt box (single serialized chain this time): raw P1 jsonl records (25-trial A/B, noise, stability, ncusolo), NCU .ncu-rep files, and the promote-pass serving runs. The pre-release promote evidence retained as provisional: the in-serving per-call profile showing `oneshotArFusedNormConstKernel` live at 8.0 us/call (800 calls; generic fallback 272 calls @ 8.0 us) — measured against the correctly-built ON+OPT server; the interleaved-chain official (293 tok/s) was contention-contaminated and is DISCARDED.
- Rebuild recipe executed: weights `/cluster-storage/shared/hf_cache/glm52-fp8` -> `/scratch/models/glm52-fp8`; `/sgl-workspace/sglang` checkout `87992eeec` + `main_port_full.diff` + `pip install -e python/ --no-deps` + `sgl-deep-gemm==0.1.4`; blog_bench restored from `/personal/glm52_backup_20260710/glm52_blog_bench`. New-box baseline text record re-captured before any gated comparison (cross-box text identity is not assumed).
- Process-hygiene lessons folded into the chain design: ONE serialized detached script (no interleaving chains), build-fingerprint gates before serving runs, quoting-proof staged scripts.

## Round 1 — route closure + block-arrival adoption (2026-07-10)

- PDL-with-predecessor probe (`bench/ar_harness.py --mode pdlprobe`): per-round
  [copy-predecessor -> AR] captured graphs, pdl=0 vs pdl=1 pairs, 10 trials,
  jit + opt, T=6/T=1. First run's jsonl records were LOST to a repo->box tar
  sync that overwrote box `bench/results.jsonl` (caught by the independent
  re-review); probe RERUN in a serialized chain — fresh records live in
  `bench/results.jsonl` (deltas +0.03%..+0.21%, inside noise, bit-ok).
  Standing rule: syncs exclude `results.jsonl`.
- Fence/flag granularity: block-arrival transform (`...NormConstKernelBA`,
  `__syncthreads()` + per-block red.async arrival; rotation counts blocks;
  exact-fit geometry asserted) taken through the FULL ladder from scratch:
  randn + value-zoo bit-exactness, hardened stability, 25-trial same-session
  A/B (1.0563/1.0884, geomean 1.0722), then a fresh serialized promote chain:
  warm 381.54 -> sanity-2 380.52 -> OFFICIAL 3x40 = 381.42 tok/s, every run
  40/40 byte-identical; restore verified 378.14 (40/40). ADOPTED into
  `oneshotArFusedConstDispatch` (cluster-arrival kernel kept as the guarded
  non-exact-fit fallback). In-serving per-call 7.6 us over 13,600 calls
  (`docs/serving_runs/percall_profile_task01_ba_tp0.txt`).
- Epilogue-fold feasibility: bounded profile of the adopted server; kernel
  adjacency extracted (`docs/profiler_evidence/trace_adjacency_task01_ba.txt`,
  method script alongside): AR followers are closed-source nvjet cuBLAS GEMMs
  (13,260/13,600) + router_gemm (340) — no post-AR quant kernel under
  bf16-dense. NO-GO recorded with conditional revisit trigger.
- Evidence plumbing from the combined re-review: adopted-entry value-zoo run
  archived (`docs/profiler_evidence/zoo_fi_vs_opt_adopted.txt`, 0 mismatched
  elements), baseline-official 376.06 summary copied to
  `docs/serving_runs/baseline_official_pretask/`, dispatch/docs references
  reconciled to the adopted kernel, `port_opt_cuh` added to harness
  provenance source_hashes.

## Round 2 — provenance closure (2026-07-10)

- `docs/baseline_source.md` runtime-path paragraph rewritten to the
  AOT-wheel reality (loader resolves to `flashinfer-jit-cache 0.6.12+cu130`,
  fast-math build; header = source-lineage artifact, binary = numeric
  reference); repo-wide stale-wording sweep clean; one leftover inline
  comment in `baseline/fi_original.py` fixed the same way.
- AC-7 evidence RERUN on box with the current harness (which now hashes
  `mnnvl_ar_fused_opt.cuh` as `port_opt_cuh` in every record's provenance):
  serialized chain `r2a_chain.sh` = 25-trial bench jit,opt + noise +
  hardened stability (pdl=1, 1000 replays = 50,000 instrumented rounds/row)
  + pdlprobe (10 trials). Results reproduce round 1: A/B 1.0546/1.0888
  geomean 1.0716 (round-1: 1.0563/1.0884/1.0722, deltas <= 0.2%), stability
  0 mismatches, pdlprobe deltas −0.18%..+0.28% inside noise, all bit-ok.
  Chain log records pre/post `results.jsonl` line-count guard (16 -> 25)
  and an in-chain provenance check: all 9 appended records carry
  `port_opt_cuh` (PROVENANCE_CHECK: PASS); log archived at
  `docs/profiler_evidence/r2a_chain_log.txt`. Records pulled box->repo
  (pull-only per BL-20260710-sync-overwrites-onbox-records).
- Independent verification re-review (Codex gpt-5.5 xhigh): both round-1
  gaps confirmed closed against the actual files/records; AC-7 gates
  re-verified on the fresh records (T=6 +5.46% vs 1.60% same-session noise,
  T=1 +8.88% vs 0.33%, geomean +7.16%; no row < 0.97x); REQUIRED_CHANGES
  empty.
