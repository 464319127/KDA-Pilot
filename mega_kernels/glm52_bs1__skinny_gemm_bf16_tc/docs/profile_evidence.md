# Profile evidence — glm52_bs1__skinny_gemm_bf16_tc

Dense bf16 projections ~2.4 ms/round (~14%) at M<=8, all dispatched by
cuBLAS to nvjet_sm103 kernels. Shape set is fixed (see config.toml). A tuned
CUDA-core GEMV (docs/prior_art/skinny_gemv.cu, rows-per-block x threads sweep
via bench/gemv_sweep.py) lost to nvjet at every shape at M=4 — the remaining
headroom is tensor-core scheduling + split-K reduce fusion + PDL chaining,
not more CUDA-core tuning.

Record the cuBLAS per-shape baseline table (CUDA-graph timed) into
docs/benchmark_method.md before optimizing; it is the contract.
