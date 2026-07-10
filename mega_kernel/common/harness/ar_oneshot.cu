// 03: single-node 8-rank one-shot push allreduce (+ residual add + rmsnorm),
// bs=1 low-latency (payload 6x6144 bf16 = 73.7KB). CUDA-graph capturable:
// every replay advances a device-side epoch counter — no host flag plumbing.
//
// Layout per rank r (workspace exchanged via P2P device pointers):
//   ws[r] = { bf16 slots[8][T*H]; uint32 flags[8]; uint32 epoch; }
// Kernel on rank r:
//   1. grid-stride copy x_r -> ws[p].slots[r] for all p (push, NVLink write)
//   2. __threadfence_system, then block0/lane<8 writes flags: ws[p].flags[r]=e
//   3. all blocks spin on ws[r].flags[p] >= e for p in 0..7
//   4. out = rmsnorm( (sum_p slots[p]) + residual ) * gamma   (fp32 accum)
// Epoch e = ++ws[r].epoch (atomic, done by block 0, broadcast via flag word
// itself: other blocks derive e from ws[r].epoch read-after-launch order —
// simpler: every block computes e = epoch_base + replay counter? No: block 0
// increments; all blocks spin until local epoch > their snapshot... To keep
// it simple and replay-safe, ALL blocks call atomicAdd on a per-block-counted
// arrival and the epoch is derived as arrivals/gridDim. Simplest correct:
// one atomicAdd per block on ws[r].epoch; e = 1 + (val / gridDim.x).
//
// This is a TASK PROTOTYPE for the isolated bench (single process, 8 devices,
// peer access enabled). Serving integration would exchange pointers via IPC
// handles like sgl custom_all_reduce.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace {

constexpr int kMaxRanks = 8;

struct Workspace {
  // slots: [kMaxRanks][max_elems] bf16, then flags[kMaxRanks], then epoch.
  // laid out by host; kernel gets raw pointers.
  __nv_bfloat16* slots[kMaxRanks];
  uint32_t* flags;   // [kMaxRanks] written by peers
  uint32_t* epoch;   // local monotonic counter
};

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// One kernel instance per rank (launched on that rank's device/stream).
// gamma == nullptr -> plain allreduce(+add) without rmsnorm.
__global__ void __launch_bounds__(256) ar_oneshot_kernel(
    const __nv_bfloat16* __restrict__ x,        // [T*H] local input
    const __nv_bfloat16* __restrict__ residual, // [T*H] or nullptr
    const __nv_bfloat16* __restrict__ gamma,    // [H] or nullptr
    __nv_bfloat16* __restrict__ out,            // [T*H]
    __nv_bfloat16* __restrict__ res_out,        // [T*H] residual out or null
    __nv_bfloat16** __restrict__ peer_slot_mine, // [world] -> ws[p].slots[my]
    volatile uint32_t** __restrict__ peer_flag_mine, // [world] -> &ws[p].flags[my]
    const __nv_bfloat16* const* __restrict__ my_slots, // [world] ws[my].slots[p]
    volatile uint32_t* __restrict__ my_flags,   // ws[my].flags
    uint32_t* __restrict__ my_epoch,
    int world, int T, int H, float eps) {
  const int total = T * H;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthreads = gridDim.x * blockDim.x;

  // ---- epoch for this replay (block-uniform, replay-monotonic) ----
  __shared__ uint32_t s_epoch;
  if (threadIdx.x == 0) {
    // Every block adds 1; epoch value for this replay = 1 + old/gridDim.
    uint32_t old = atomicAdd(my_epoch, 1u);
    s_epoch = 1u + old / (gridDim.x * gridDim.y);
  }
  __syncthreads();
  const uint32_t e = s_epoch;

  // ---- push my shard to every rank's slot[my] (uint4 = 8 bf16) ----
  const uint4* xv = reinterpret_cast<const uint4*>(x);
  const int vec_total = total / 8;
  if (blockIdx.y == 0) {  // push once, not once per token-block
    for (int p = 0; p < world; ++p) {
      uint4* dst = reinterpret_cast<uint4*>(peer_slot_mine[p]);
      for (int i = tid; i < vec_total; i += nthreads) dst[i] = xv[i];
    }
    __threadfence_system();
    if (blockIdx.x == 0 && threadIdx.x < (unsigned)world) {
      *peer_flag_mine[threadIdx.x] = e;
    }
  }
  // ---- wait for all ranks' shards ----
  if (threadIdx.x < (unsigned)world) {
    while (my_flags[threadIdx.x] < e) { }
  }
  __syncthreads();
  __threadfence_system();  // acquire: order peer-data reads after flag observation

  // ---- reduce + add + rmsnorm ----
  // One block per token owns the whole row (blockIdx.x == 0 of each y-slice):
  // 24 redundant x-blocks racing on the same out[] corrupted the norm phase
  // in v1 — the extra x-blocks exist only to widen the push, so they exit here.
  const int t = blockIdx.y;
  if (t >= T || blockIdx.x != 0) return;
  const int base = t * H;
  float local_sq = 0.f;
  extern __shared__ float s_red[];
  // each thread processes H/blockDim elements
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float acc = 0.f;
#pragma unroll
    for (int p = 0; p < kMaxRanks; ++p) {
      if (p < world) acc += bf16_to_f32(my_slots[p][base + h]);
    }
    if (residual != nullptr) acc += bf16_to_f32(residual[base + h]);
    if (res_out != nullptr) res_out[base + h] = __float2bfloat16(acc);
    // stash acc for pass 2 in out (pre-norm) — reread below
    out[base + h] = __float2bfloat16(acc);
    local_sq += acc * acc;
  }
  // block reduce
  s_red[threadIdx.x] = local_sq;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) s_red[threadIdx.x] += s_red[threadIdx.x + s];
    __syncthreads();
  }
  if (gamma == nullptr) return;  // plain AR(+add)
  const float rms = rsqrtf(s_red[0] / H + eps);
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float v = bf16_to_f32(out[base + h]);
    out[base + h] = __float2bfloat16(v * rms * bf16_to_f32(gamma[h]));
  }
}

}  // namespace

// Host-side: tensors created per rank by the harness; pointer tables are
// int64 tensors of raw device addresses (prepared once after P2P enable).
void ar_oneshot(torch::Tensor x, torch::Tensor residual, torch::Tensor gamma,
                torch::Tensor out, torch::Tensor res_out,
                torch::Tensor peer_slot_mine, torch::Tensor peer_flag_mine,
                torch::Tensor my_slots, torch::Tensor my_flags,
                torch::Tensor my_epoch, int64_t world, int64_t T, int64_t H,
                double eps, int64_t grid_x) {
  const c10::cuda::CUDAGuard g(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)grid_x, (unsigned)T);
  dim3 block(256);
  size_t smem = 256 * sizeof(float);
  ar_oneshot_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      residual.numel() ? reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()) : nullptr,
      gamma.numel() ? reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()) : nullptr,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      res_out.numel() ? reinterpret_cast<__nv_bfloat16*>(res_out.data_ptr()) : nullptr,
      reinterpret_cast<__nv_bfloat16**>(peer_slot_mine.data_ptr()),
      reinterpret_cast<volatile uint32_t**>(peer_flag_mine.data_ptr()),
      reinterpret_cast<const __nv_bfloat16* const*>(my_slots.data_ptr()),
      reinterpret_cast<volatile uint32_t*>(my_flags.data_ptr()),
      reinterpret_cast<uint32_t*>(my_epoch.data_ptr()),
      (int)world, (int)T, (int)H, (float)eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ar_oneshot", &ar_oneshot, "8-rank oneshot AR(+add+rmsnorm), graph-capturable");
}
