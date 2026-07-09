// K2v4: tensor-core small-M GEMM for w8a8 block-FP8 dense linears (bs=1 MTP).
// out[M,N](bf16) = A[M,K](bf16) @ dequant(W[N,K] fp8e4m3, S[N/128,K/128] f32)^T
// M <= 8 (padded to 16 for mma.m16n8k16).
//
// Layout per CTA (128 threads = 4 warps):
//   BLOCK_N = 64 columns of W (rows of W matrix), full K sweep in 128-wide steps.
//   warp w owns n-slice [w*16, w*16+16) -> two n8 mma tiles.
//   Per k-step (BLOCK_K=128): 8 mma.m16n8k16 per n8 tile (k chunks of 16).
// W is staged via cp.async (8KB/step, double buffered); A tile (16x128 f16,
// 4KB) is converted from bf16 gmem each step by all warps cooperatively into
// smem. The per-(128n,128k) scale is folded into the fp8->f16 dequant so mma
// accumulates across the whole K sweep in fp32 fragments.
// Grid: (ceil(N/64), SPLIT_K). Deterministic split-K: fp32 partials + reduce
// pass (k2_gemv_pass2-compatible layout [S, M_PAD, N]).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#define M_PAD 16
#define BLOCK_N 64
#define BLOCK_K 128

// ldmatrix helpers -----------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2],
                                                  uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
      : "=r"(r[0]), "=r"(r[1])
      : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16_f16f32(float (&d)[4],
                                                    const uint32_t (&a)[4],
                                                    const uint32_t (&b)[2],
                                                    const float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                   smem_u32addr(smem)),
               "l"(gmem));
}

// fp8x2 (e4m3) -> f16x2 hardware convert
__device__ __forceinline__ uint32_t cvt_e4m3x2_f16x2(uint16_t x) {
  uint32_t out;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(x));
  return out;
}

extern "C" __global__ __launch_bounds__(128, 4) void k2_mma_pass1(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const uint8_t* __restrict__ W,        // [N, K] fp8 e4m3 row-major
    const float* __restrict__ S,          // [ceil(N/128), ceil(K/128)]
    float* __restrict__ P,   // fp32 partials [SPLIT_K, M_PAD, N]; or bf16 out
    __nv_bfloat16* __restrict__ O,  // [M, N] used when SPLIT_K == 1
    int M, int N, int K, int KB, int kb_per_split, int s_stride,
    int p_stride_split, int p_stride_m, int split_k) {
  const int n0 = blockIdx.x * BLOCK_N;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  const int kb0 = blockIdx.y * kb_per_split;
  const int kb1 = min(kb0 + kb_per_split, KB);

  // Smem: W staged as f16 after dequant? No — stage RAW fp8, convert during
  // ldmatrix? ldmatrix needs 16-bit. Instead: stage fp8 raw (8KB), each warp
  // dequants its n16 slice into an f16 smem tile (with scale folded), then
  // ldmatrix from there. A: staged bf16->f16 (4KB).
  // buffers: wraw[2][64][128] u8 (16KB), wf16[64][128+8] f16 for CURRENT step
  // per warp slice (we dequant per step; single buffer, 16KB+pad), af16[16][128+8].
  extern __shared__ uint8_t smem[];
  uint8_t* wraw0 = smem;                       // 8KB
  uint8_t* wraw1 = smem + 8 * 1024;            // 8KB
  __half* wf16 = reinterpret_cast<__half*>(smem + 16 * 1024);  // 64*136*2=17408
  __half* af16 = reinterpret_cast<__half*>(smem + 16 * 1024 + 17408);  // 16*136*2

  const int WROW_PITCH = 128;       // raw fp8 row pitch
  const int F16_PITCH = 136;        // padded to avoid bank conflicts

  // ---- cp.async W tile loader: 64 rows x 128B; 128 threads x (4 x 16B) ----
  auto load_w = [&](uint8_t* dst, int kb) {
    const int k0 = kb << 7;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int slot = threadIdx.x + i * 128;          // 0..511
      int row = slot >> 3;                       // 64 rows, 8 slots/row
      int col16 = slot & 7;                      // 8 x 16B = 128B
      int gn = n0 + row;
      const void* src = W + (size_t)gn * K + k0 + col16 * 16;
      if (gn < N)
        cp_async_16(dst + row * WROW_PITCH + col16 * 16, src);
    }
    asm volatile("cp.async.commit_group;");
  };

  // ---- A tile: 16 x 128 bf16 -> f16 smem (uniform across CTAs) ----
  auto load_a = [&](int kb) {
    const int k0 = kb << 7;
    // 16*128 = 2048 elems; 128 threads x 16 elems
    for (int i = threadIdx.x * 16; i < M_PAD * 128; i += 128 * 16) {
      int r = i >> 7;          // /128
      int c = i & 127;
#pragma unroll
      for (int j = 0; j < 16; j += 2) {
        int cc = c + j;
        __half2 h2;
        if (r < M) {
          float f0 = __bfloat162float(A[(size_t)r * K + k0 + cc]);
          float f1 = __bfloat162float(A[(size_t)r * K + k0 + cc + 1]);
          h2 = __floats2half2_rn(f0, f1);
        } else {
          h2 = __floats2half2_rn(0.f, 0.f);
        }
        *reinterpret_cast<__half2*>(af16 + r * F16_PITCH + cc) = h2;
      }
    }
  };

  // fragments: each warp: 2 n8-tiles, acc[2][4]
  float acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};

  int buf = 0;
  load_w(wraw0, kb0);

  for (int kb = kb0; kb < kb1; ++kb) {
    // prefetch next W
    if (kb + 1 < kb1) load_w(buf ? wraw0 : wraw1, kb + 1);

    load_a(kb);  // overlaps with cp.async in flight

    asm volatile("cp.async.wait_group 1;");  // current W ready (next may fly)
    __syncthreads();

    const uint8_t* wr = buf ? wraw1 : wraw0;
    const float s_f32 = S[(n0 >> 7) * s_stride + kb];
    const __half s_h = __float2half_rn(s_f32);
    const __half2 s_h2 = __halves2half2(s_h, s_h);

    // dequant own n16 slice: rows [warp*16, warp*16+16), 128 cols fp8
    {
      const int r0 = warp * 16;
      // 16 rows x 128 cols = 2048 fp8; 32 lanes x 64 = per lane 64 vals
      for (int i = lane * 4; i < 16 * 128; i += 32 * 4) {
        int r = i >> 7;
        int c = i & 127;
        const uint8_t* src = wr + (r0 + r) * WROW_PITCH + c;
        uint32_t packed4;
        memcpy(&packed4, src, 4);
        uint32_t lo = cvt_e4m3x2_f16x2((uint16_t)(packed4 & 0xffff));
        uint32_t hi = cvt_e4m3x2_f16x2((uint16_t)(packed4 >> 16));
        __half2 lo2 = __hmul2(*reinterpret_cast<__half2*>(&lo), s_h2);
        __half2 hi2 = __hmul2(*reinterpret_cast<__half2*>(&hi), s_h2);
        *reinterpret_cast<__half2*>(wf16 + (r0 + r) * F16_PITCH + c) = lo2;
        *reinterpret_cast<__half2*>(wf16 + (r0 + r) * F16_PITCH + c + 2) = hi2;
      }
    }
    __syncthreads();

    // mma over 8 k16 chunks
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      // A frag m16k16: ldmatrix x4 from af16
      uint32_t afrag[4];
      {
        int row = lane & 15;
        int half_sel = lane >> 4;  // 0/1 -> k half
        uint32_t addr = smem_u32addr(af16 + row * F16_PITCH + kk * 16 +
                                     half_sel * 8);
        ldmatrix_x4(afrag, addr);
      }
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        // B frag k16n8 from wf16 rows [warp*16 + t*8, +8) — column-major for
        // mma "col": use ldmatrix x2 trans on the 8 rows x 16 cols block.
        uint32_t bfrag[2];
        int row = warp * 16 + t * 8 + (lane & 7);
        int seg = (lane >> 3) & 1;  // two 8-col halves
        uint32_t addr =
            smem_u32addr(wf16 + row * F16_PITCH + kk * 16 + seg * 8);
        ldmatrix_x2_trans(bfrag, addr);
        mma_m16n8k16_f16f32(acc[t], afrag, bfrag, acc[t]);
      }
    }
    __syncthreads();
    buf ^= 1;
  }

  // ---- epilogue: write fragments ----
  // mma m16n8 output layout: lane l holds c[0..3]:
  //   row = (l >> 2) + 8*(i>=2 ? 1 : 0)?? standard: c0,c1 -> rows l/4, cols
  //   2*(l%4)+ {0,1}; c2,c3 -> rows l/4+8, same cols.
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    int ncol0 = n0 + warp * 16 + t * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int r = (lane >> 2) + (i >= 2 ? 8 : 0);
      int c = ncol0 + 2 * (lane & 3) + (i & 1);
      if (c < N && r < M_PAD) {
        if (split_k == 1) {
          if (r < M) O[(size_t)r * N + c] = __float2bfloat16(acc[t][i]);
        } else {
          P[(size_t)blockIdx.y * p_stride_split + (size_t)r * p_stride_m + c] =
              acc[t][i];
        }
      }
    }
  }
}

// ---- split-K reduce: [SPLIT_K, M_PAD, N] fp32 -> [M, N] bf16 ----
extern "C" __global__ __launch_bounds__(256, 8) void k2_mma_pass2(
    const float* __restrict__ P, __nv_bfloat16* __restrict__ O, int M, int N,
    int split_k, int p_stride_split, int p_stride_m) {
  const int n = blockIdx.x * 256 + threadIdx.x;
  if (n >= N) return;
#pragma unroll 4
  for (int m = 0; m < M; ++m) {
    float v = 0.f;
    for (int s = 0; s < split_k; ++s)
      v += P[(size_t)s * p_stride_split + (size_t)m * p_stride_m + n];
    O[(size_t)m * N + n] = __float2bfloat16(v);
  }
}

// ---- torch extension bindings ----
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static constexpr int kSmemBytes = 16 * 1024 + 64 * 136 * 2 + 16 * 136 * 2;

void pass1(at::Tensor A, at::Tensor W, at::Tensor S, at::Tensor P, at::Tensor O,
           int64_t M, int64_t N, int64_t K, int64_t KB, int64_t kb_per_split,
           int64_t s_stride, int64_t p_stride_split, int64_t p_stride_m,
           int64_t split_k) {
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, split_k);
  dim3 block(128);
  auto stream = at::cuda::getCurrentCUDAStream();
  k2_mma_pass1<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
      reinterpret_cast<const uint8_t*>(W.data_ptr()),
      S.data_ptr<float>(),
      P.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
      (int)M, (int)N, (int)K, (int)KB, (int)kb_per_split, (int)s_stride,
      (int)p_stride_split, (int)p_stride_m, (int)split_k);
}

void pass2(at::Tensor P, at::Tensor O, int64_t M, int64_t N, int64_t split_k,
           int64_t p_stride_split, int64_t p_stride_m) {
  dim3 grid((N + 255) / 256);
  dim3 block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  k2_mma_pass2<<<grid, block, 0, stream>>>(
      P.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
      (int)M, (int)N, (int)split_k, (int)p_stride_split, (int)p_stride_m);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pass1", &pass1);
  m.def("pass2", &pass2);
}
