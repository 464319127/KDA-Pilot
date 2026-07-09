# integration — promoted kernels back into serving

Env-gated hooks already established in the GLM-5.2 bs=1 patch set
(`glm52_blog_bench/patches/`, applies to sglang v0.5.14 checkout `49e384c`):

| task | hook point | env gate |
|---|---|---|
| 01 dense GEMM | `fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback` M≤8 dispatch; float scales stashed on weights at load via `requant_weight_ue8m0` | `SGLANG_BS1_TRITON_FP8_GEMV` slot (rename) |
| 02 MoE aux | `moe_runner/flashinfer_trtllm.py` fp8 branch (defer-finalize pattern) | `SGLANG_BS1_FP8_DEFER_FINALIZE` precedent |
| 03 AR+norm | `communicator.py` / `flashinfer_comm_fusion.py` fusion backend option | new backend name |
| 04 attention | `dsa_backend._forward_trtllm` decode impl switch | `--dsa-decode-backend` style |
| 05 PDL | capture-time launch attrs per kernel library | per-lib flags |

Validation ladder (devbox `glm52-bs1-opt`, same node as all baselines):
1. task-level correctness + cold-L2 bench (task dir)
2. dummy-weight 8-layer server smoke (`launch_dummy.sh`, ~3 min startup)
3. real-weight 1×40 sanity: greedy determinism + accept-len (3.95±0.02) +
   decode tok/s
4. official 3×40 run for the record (REPORT_OPT.md)

Current official chain: 307.17 → 316.05 → 354.10 tok/s. Iteration budget
tracker: GPU 9.3 ms + CPU exposure 1.8 ms = 11.15 ms @ accept 3.95.
