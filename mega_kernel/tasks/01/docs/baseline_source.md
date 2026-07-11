# Baseline Source Provenance — mega_kernel task 01

## Baseline identity (documented deviation D4)

The kernel under comparison does not exist in sglang upstream; sglang is the port DESTINATION. The baseline/oracle is the flashinfer MNNVL fused allreduce kernel as installed on the target devbox.

## Kernel source (verbatim copy)

- Upstream package: flashinfer **0.6.12** (installed wheel on devbox `glm52-bs1-opt`, worker `light-face-hides-fin-03-1`)
- Source path on box: `/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh` (the header flashinfer ships for JIT builds, 1223 lines; NOTE: on this box the loader resolves to the precompiled AOT module — see "Baseline runner ABI and executed binary" below — so this header is the source-lineage artifact, not the executed binary)
- Copied to: `baseline/trtllm_mnnvl_allreduce.cuh`
- sha256 (box == local copy, verified): `ab6560f28bf94d06ce5ee80bd8414674d9d82c72fba6c8a19454b9c6ea1297aa`
- Copy time: 2026-07-10T08:28:41Z (box clock, UTC)
- Local edits: none (verbatim; any future edit would break the baseline contract)
- Target symbol: `flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<WorldSize, T, RMSNormFusion, PackedType>` + `oneshotAllreduceFusionDispatch` (the header's twoshot path is not the bs=1 target)

## Reference copies (context only — NOT runtime dependencies)

Located under `baseline/upstream_ref/`; the harness drives the ORIGINAL kernel through the installed flashinfer package, never through these copies:

- `flashinfer/trtllm_mnnvl_ar.py` — `MNNVLAllReduceFusionWorkspace` (lamport buffers, `buffer_flags`, `mc_ptr`), FFI op `trtllm_mnnvl_allreduce_fusion(...)`, strategy selection
- `flashinfer/mnnvl.py` — `MnnvlMemory` fabric/multicast allocation (cuMulticast create/bind, 512 MB fabric pages), comm-backend abstraction
- `flashinfer/workspace_base.py` — `AllReduceFusionWorkspace` base
- `sglang/flashinfer_comm_fusion.py` — serving callsite (`flashinfer_allreduce_residual_rmsnorm` custom op; mnnvl backend auto-selection on SM103) from the serving checkout
- `sglang/moe_finalize_fuse_shared.py` — sglang jit_kernel module pattern reference (load/compile usage) from the serving checkout

## Upstream lineage (KernelWiki)

- `external/KernelWiki/sources/prs/flashinfer/PR-1321.md` — "Optimizations for TRTLLM MNNVL Allreduce" (sm100; touches this header)
- `external/KernelWiki/sources/prs/flashinfer/PR-2118.md` — "Refactor trtllm_mnnvl_allreduce"; adds `trtllm_mnnvl_allreduce` / fused-AR + rmsnorm API surface
- `external/KernelWiki/sources/prs/flashinfer/PR-2130.md` — unified allreduce-fusion workspace API (`create_allreduce_fusion_workspace` / `allreduce_fusion`, backend `trtllm|mnnvl|auto`)

## Serving destination pin (integration context, not kernel baseline)

- sglang repo: https://github.com/sgl-project/sglang, branch `main`
- Serving checkout on box: `/sgl-workspace/sglang` at `87992eeec4072995e8fa98fb2d0f3a7e5e581f2d` ("[DeepSeek V2] Reorder dual-stream MoE to main-first to avoid CUDA graph stream explosion (#30460)") — verified 2026-07-10; plus `/personal/glm52_backup_20260710/patches_main/main_port_full.diff` and `sgl-deep-gemm==0.1.4` per the documented serving rebuild recipe.
- Resolution time: 2026-07-10T08:28:41Z

## Baseline runner ABI and executed binary (corrected after the round-0 AOT-wheel discovery)

`baseline/fi_original.py` (created with the harness) exposes the flashinfer ORIGINAL through the same low-overhead local entry ABI as the candidate: identical argument order, stream semantics, preallocated outputs, and shared NVLS multimem workspace.

Runtime path — what actually executes: the installed flashinfer package's module loader (`get_library_path()`) resolves to the PRECOMPILED `flashinfer-jit-cache 0.6.12+cu130` AOT module on this box; there is no locally JIT-built module, so the AOT wheel binary IS the production baseline the harness and serving both run. That binary is a fast-math build (SASS fingerprint: 96,240 `.FTZ` instruction modifiers, MUFU.RCP approximate division — see `docs/run_log.md` "Flag-ON Divergence Diagnosis"). It was verified against the copied header: identical kernel symbol signatures (same template surface incl. the weightBias parameter), consistent with being built from the same source revision as `baseline/trtllm_mnnvl_allreduce.cuh`. The copied header remains the SOURCE-LINEAGE artifact (what the port's source is verbatim-diffed against); the AOT binary is the NUMERIC reference (which is why the port builds with baseline-matching `--use_fast_math`).

Negative guarantee (unchanged): no imports from the live serving checkout and no imports from `baseline/upstream_ref/` at benchmark time.
