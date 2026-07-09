// DIAGNOSTIC native-CUDA block-scaled FP8 GEMM for the M>1 regimes of the
// deep_gemm_fp8_fp8_bf16_nt interface (M==113 small-prefill rows and tiny-K
// K in {256, 512} rows). B200 (SM100).
//
// Purpose: this is the bounded CUDA-core candidate ATTEMPT for the M>1 buckets.
// The pre-attempt analysis predicts it loses to DeepGEMM's tensor-core kernel by
// several times (CUDA-core FMA ceiling ~64-128 TFLOP/s vs DeepGEMM's measured
// 0.4-2.45 PFLOP/s on these rows); it exists to turn that structural argument into
// measured per-row numbers under the task's fair harness. It is NOT promoted by
// default: `solution/candidate_entry.py` routes to it only when the measurement
// flag KDA_DIAG_NATIVE_MGT1=1 is set, and only for its exact predicate.
//
// Semantics are identical to the recovered baseline (NT, per-128-K-block 1Dx1D
// UE8M0 scaling; same packed int32 MN-major scale decode as decode_gemv.cu):
//   out[m, n] = sum_k float(A[m,k])*scale_a(m,k/128) * float(B[n,k])*scale_b(n,k/128)
//
// Structure: one warp owns one output column n (coalesced 512 B B reads per step,
// register-shuffled packed B-scale words), while the CTA cooperatively stages an
// 8-row A tile chunk (TM x 512 B) plus its dequantized per-group scales in shared
// memory per (m-tile, k-step); each lane carries TM fp32 accumulators so the m-loop
// reuses the streamed B chunk TM times. Rows beyond the tile/tail are zero-filled
// in the staging so the accumulation needs no per-row branches.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "fp8_gemv_common.cuh"

namespace {

constexpr int kWarpsPerCta = 4;
constexpr int kThreads = kWarpsPerCta * 32;
constexpr int kTileRows = 8;  // fp32 accumulators per lane


__global__ void __launch_bounds__(kThreads) mtile_gemm_diag_kernel(
    const uint8_t* __restrict__ A,     // [M, K] row-major
    const int32_t* __restrict__ Asf,   // packed UE8M0, MN-major [M, W]
    const uint8_t* __restrict__ B,     // [N, K] row-major
    const int32_t* __restrict__ Bsf,   // packed UE8M0, MN-major [N, W]
    __nv_bfloat16* __restrict__ out,   // [M, N] row-major
    int M, int N, int K,
    int Asf_s1, int Bsf_s0, int Bsf_s1) {
    extern __shared__ unsigned char smem[];
    uint8_t* sA = smem;                                        // kTileRows x 512 bytes
    float* sAscale = reinterpret_cast<float*>(sA + kTileRows * 512);  // kTileRows x 4

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int num_groups = K >> 7;
    const int num_steps = (num_groups + 3) >> 2;

    const int n = blockIdx.x * kWarpsPerCta + warp;
    const bool valid_col = n < N;
    const uint8_t* __restrict__ Brow = B + static_cast<size_t>(valid_col ? n : 0) * K;
    const int32_t* __restrict__ BsfRow = Bsf + static_cast<size_t>(valid_col ? n : 0) * Bsf_s0;

    // All packed B-scale words this column can need (word index == step, <=12).
    const uint32_t my_word = (valid_col && lane < num_steps)
        ? static_cast<uint32_t>(BsfRow[lane * Bsf_s1]) : 0u;

    for (int m0 = 0; m0 < M; m0 += kTileRows) {
        const int tile_rows = min(kTileRows, M - m0);
        float acc[kTileRows];
#pragma unroll
        for (int r = 0; r < kTileRows; ++r) acc[r] = 0.0f;

        for (int step = 0; step < num_steps; ++step) {
            __syncthreads();  // previous step's tile fully consumed
            // Stage the A tile chunk; rows beyond the tail and bytes beyond K are
            // zero-filled so the compute loop needs no per-row/tail branches.
            for (int i = tid * 16; i < kTileRows * 512; i += kThreads * 16) {
                const int r = i >> 9;
                const int off_in_row = i & 511;
                const int gb = step * 512 + off_in_row;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (r < tile_rows && gb < K) {
                    v = *reinterpret_cast<const uint4*>(A + static_cast<size_t>(m0 + r) * K + gb);
                }
                *reinterpret_cast<uint4*>(sA + i) = v;
            }
            if (tid < kTileRows * 4) {
                const int r = tid >> 2;
                const int g = tid & 3;
                const int kg = step * 4 + g;
                float s = 0.0f;
                if (r < tile_rows && kg < num_groups) {
                    const uint32_t w = static_cast<uint32_t>(Asf[(m0 + r) + step * Asf_s1]);
                    s = ue8m0_byte_to_scale(w, g);
                }
                sAscale[tid] = s;
            }
            __syncthreads();

            // Warp-uniform: every lane of every warp executes this shuffle at the
            // same call site (the K tail, e.g. K == 256, diverges lanes below).
            const uint32_t b_word = __shfl_sync(0xFFFFFFFFu, my_word, step);
            const int off = step * 512 + lane * 16;
            if (valid_col && off < K) {
                const float sb = ue8m0_byte_to_scale(b_word, lane >> 3);
                const uint4 bw = *reinterpret_cast<const uint4*>(Brow + off);
#pragma unroll
                for (int r = 0; r < kTileRows; ++r) {
                    const uint4 aw = *reinterpret_cast<const uint4*>(sA + r * 512 + lane * 16);
                    acc[r] += dot16_fp8(aw, bw) * sAscale[r * 4 + (lane >> 3)] * sb;
                }
            }
        }

#pragma unroll
        for (int r = 0; r < kTileRows; ++r) {
            float v = acc[r];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
            }
            if (valid_col && lane == 0 && r < tile_rows) {
                out[static_cast<size_t>(m0 + r) * N + n] = __float2bfloat16(v);
            }
        }
    }
}

}  // namespace

// Diagnostic launcher (current CUDA stream, destination passing). Layout gating
// lives in the Python-side predicate; only cheap structural invariants here.
void mtile_gemm_diag(torch::Tensor A, torch::Tensor Asf,
                     torch::Tensor B, torch::Tensor Bsf,
                     torch::Tensor out) {
    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));
    TORCH_CHECK(M > 1 && (K & 127) == 0, "mtile_gemm_diag: M must be > 1, K a multiple of 128");
    TORCH_CHECK(((K >> 7) + 3) / 4 <= 32,
                "mtile_gemm_diag: at most 32 packed scale words supported (K <= 16384); the\n"
                "warp-shuffle scale preload selects words by source lane");
    TORCH_CHECK(B.size(1) == K && out.size(0) == M && out.size(1) == N,
                "mtile_gemm_diag: shape mismatch");
    TORCH_CHECK(Asf.get_device() == A.get_device() && B.get_device() == A.get_device()
                    && Bsf.get_device() == A.get_device() && out.get_device() == A.get_device(),
                "mtile_gemm_diag: all tensors must be on one CUDA device");

    // In a multi-GPU process the current device may differ from the tensors'
    // device; the stream below must belong to the tensors' device.
    const c10::cuda::CUDAGuard device_guard(A.device());
    const int blocks = (N + kWarpsPerCta - 1) / kWarpsPerCta;
    const size_t smem = static_cast<size_t>(kTileRows) * 512 + kTileRows * 4 * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    mtile_gemm_diag_kernel<<<blocks, kThreads, smem, stream>>>(
        reinterpret_cast<const uint8_t*>(A.data_ptr()),
        Asf.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t*>(B.data_ptr()),
        Bsf.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        M, N, K,
        static_cast<int>(Asf.stride(1)),
        static_cast<int>(Bsf.stride(0)), static_cast<int>(Bsf.stride(1)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mtile_gemm_diag", &mtile_gemm_diag,
          "Diagnostic M>1 FP8 block-scaled CUDA-core GEMM (B200/SM100)");
}
