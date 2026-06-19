# TileRT kernel correctness contract

Each candidate CUDA kernel must match its `baseline/*.py` reference (which is
TileRT's own golden_forward math, validated against the real op):
- bf16 GEMM / attention outputs: relative L2 error < 2e-2.
- FP8/FP4-quantized outputs: dequantized rel L2 < 5e-2 (quantization noise).
- Same output shape/dtype/device as the baseline; destination-passing ABI.
Correctness is checked on every workload shape (seq ∈ {1,2,4}) before timing.
Do not import/patch TileRT or SGLang during correctness/benchmark; call only
files in the task dir.
