// Bit-exact split-RoPE pair kernel for LTX2 (NVIDIA B200 / sm_100).
//
// Reproduces the eager apply_split_rotary_emb rounding sequence exactly (see
// docs/numerics_characterization.md, variant A1):
//   c1 = bf16(first * cos)                       // first product rounded to bf16
//   out_first  = bf16( c1 - sin*second )         // sine term combined in fp32, single cast
//   out_second = bf16( bf16(second*cos) + sin*first )
// All fp32 combines use explicit rounded intrinsics so -O3 cannot contract them
// into a fused multiply-add (which would change the visible rounding). cos/sin are
// indexed via their real (non-contiguous) strides; only the last dim is stride-1.
//
// This is the RoPE stage only; RMSNorm is applied upstream (ATen, bit-exact) and
// the normalized tensor is passed in as x. Layout: x/out are [B,S,H] with
// H = num_heads*head_dim, head-major (head h occupies [h*head_dim, (h+1)*head_dim)).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

using tvm::ffi::TensorView;

__global__ void ltx2_split_rope_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cosp,
    const __nv_bfloat16* __restrict__ sinp,
    __nv_bfloat16* __restrict__ out,
    long B, long S, long num_heads, long head_dim, long r,
    long xs0, long xs1, long xs2,
    long os0, long os1, long os2,
    long cs0, long cs1, long cs2, long cs3,
    long ss0, long ss1, long ss2, long ss3) {
  long total = B * S * num_heads * r;
  long idx = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
  if (idx >= total) return;

  long j = idx % r;
  long t = idx / r;
  long head = t % num_heads;
  t /= num_heads;
  long s = t % S;
  long b = t / S;

  long i0 = head * head_dim + j;  // first-half element within H
  long i1 = i0 + r;               // paired second-half element

  const __nv_bfloat16* xrow = x + b * xs0 + s * xs1;
  float first = __bfloat162float(xrow[i0 * xs2]);
  float second = __bfloat162float(xrow[i1 * xs2]);

  float c = __bfloat162float(cosp[b * cs0 + head * cs1 + s * cs2 + j * cs3]);
  float sn = __bfloat162float(sinp[b * ss0 + head * ss1 + s * ss2 + j * ss3]);

  // Visible first rounding: round (split_x * cos) to bf16 before the sine term.
  float c1 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(first, c)));
  float c2 = __bfloat162float(__float2bfloat16_rn(__fmul_rn(second, c)));

  // Sine term combined in fp32 with a single final bf16 cast (no FMA contraction).
  float of = __fsub_rn(c1, __fmul_rn(sn, second));  // first*cos - sin*second
  float os = __fadd_rn(c2, __fmul_rn(sn, first));    // second*cos + sin*first

  __nv_bfloat16* orow = out + b * os0 + s * os1;
  orow[i0 * os2] = __float2bfloat16_rn(of);
  orow[i1 * os2] = __float2bfloat16_rn(os);
}

void ltx2_split_rope_candidate(TensorView x, TensorView cos, TensorView sin, TensorView out) {
  // Guard the launch to the input tensor's CUDA device so the stream and the kernel launch target
  // the device the data lives on, even if it differs from the process's current device (multi-GPU).
  const c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(x.device().device_id));
  long B = x.size(0);
  long S = x.size(1);
  long num_heads = cos.size(1);
  long r = cos.size(3);
  long head_dim = 2 * r;

  long total = B * S * num_heads * r;
  if (total == 0) return;

  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const __nv_bfloat16* cp = reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr());
  const __nv_bfloat16* sp = reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr());
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

  int block = 256;
  unsigned int grid = (unsigned int)((total + block - 1) / block);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ltx2_split_rope_kernel<<<grid, block, 0, stream>>>(
      xp, cp, sp, op,
      B, S, num_heads, head_dim, r,
      x.stride(0), x.stride(1), x.stride(2),
      out.stride(0), out.stride(1), out.stride(2),
      cos.stride(0), cos.stride(1), cos.stride(2), cos.stride(3),
      sin.stride(0), sin.stride(1), sin.stride(2), sin.stride(3));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(ltx2_split_rope_candidate, ltx2_split_rope_candidate);
