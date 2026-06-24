#pragma once
#include <torch/all.h>

// SM100 swap-AB FP8 per-token/per-channel scaled GEMM for small M (in (1,64]).
// Inputs are the captured contract: A=[M,K] fp8 row-major, B=[K,N] fp8 column-major
// (Bphys[N,K]), scale_a=[M,1] fp32, scale_b=[N,1] fp32, out=[M,N] bf16 row-major
// (pre-allocated). Defined in solution/fp8_swapab_smallm.cu (own CUTLASS TU);
// called by the candidate dispatch for covered small-M shapes.
void fp8_scaled_mm_swapab_smallm(
    torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& scale_a, const torch::Tensor& scale_b);
