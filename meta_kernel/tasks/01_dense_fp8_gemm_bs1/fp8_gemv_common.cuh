// Shared device helpers for the native FP8 block-scaled kernels of the
// deep_gemm_fp8_fp8_bf16_nt task (decode_gemv.cu and mtile_gemm_diag.cu).
//
// Packed-UE8M0 decode contract (verified bit-exact against deep_gemm's
// reference packing on B200): an int32 word holds four 8-bit exponents in
// little-endian byte order; for K-group kg, word index = kg >> 2, byte lane =
// kg & 3, and scale = 2^(E-127) = __int_as_float(E << 23) (E == 0 -> 0.0f).
//
// NOTE: torch.utils.cpp_extension.load() keys its rebuild cache on the listed
// .cu sources and flags, not on included headers — after editing THIS file,
// touch the including .cu files (or clear the extension cache) to force a
// rebuild.

#pragma once

#include <cuda_fp8.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float ue8m0_byte_to_scale(uint32_t word, int byte_idx) {
    uint32_t e = (word >> (8 * byte_idx)) & 0xFFu;
    return __int_as_float(static_cast<int>(e << 23));  // 2^(e-127); e==0 -> 0.0f
}

__device__ __forceinline__ void unpack4_fp8(uint32_t w, float& f0, float& f1, float& f2, float& f3) {
    __half2 lo = __half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(w & 0xFFFFu), __NV_E4M3));
    __half2 hi = __half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>((w >> 16) & 0xFFFFu), __NV_E4M3));
    float2 lof = __half22float2(lo);
    float2 hif = __half22float2(hi);
    f0 = lof.x; f1 = lof.y; f2 = hif.x; f3 = hif.y;
}

// Dot product of 16 fp8_e4m3 pairs (one uint4 from each side), accumulated in
// fp32 via the exact fp8x2 -> half2 -> float path.
__device__ __forceinline__ float dot16_fp8(const uint4& a, const uint4& b) {
    float a0, a1, a2, a3, b0, b1, b2, b3;
    float acc = 0.0f;
    unpack4_fp8(a.x, a0, a1, a2, a3); unpack4_fp8(b.x, b0, b1, b2, b3);
    acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    unpack4_fp8(a.y, a0, a1, a2, a3); unpack4_fp8(b.y, b0, b1, b2, b3);
    acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    unpack4_fp8(a.z, a0, a1, a2, a3); unpack4_fp8(b.z, b0, b1, b2, b3);
    acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    unpack4_fp8(a.w, a0, a1, a2, a3); unpack4_fp8(b.w, b0, b1, b2, b3);
    acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    return acc;
}
