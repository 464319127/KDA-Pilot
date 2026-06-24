// Candidate: SM100 swap-AB CUTLASS FP8 per-token/per-channel scaled GEMM for the
// small-M decode regime (M in (1, 64]). Re-instantiation of the upstream
// DeviceGemmFp8RowwiseSm100 pattern (CUTLASS C++ templates are allowed candidate
// code) with the operands/scales/output transposed so the GEMM's M-dimension
// covers the large N (fully tiled) and its N-dimension covers the small original
// M (on the tolerant N-axis), avoiding the baseline's wasted 64-row CTA-M tile.
// Recipe (Codex task24 / KernelWiki pr-vllm-27284):
//   GEMM-A = original B = Bphys[N,K] row-major (b.data_ptr())
//   GEMM-B = original A = A^T[K,M] column-major (a.data_ptr())
//   problem {m',n',k'} = {N, M, K}; out^T[N,M] written column-major so D'(n,m) ->
//   out[m,n] = m*N+n (== row-major out[M,N], no transpose/copy)
//   scale swap: ScaleA(col-broadcast) <- scale_b, ScaleB(row-broadcast) <- scale_a
//   => out = scale_a[m]*scale_b[n]*acc (EVT unchanged)
//
// Compiled in its own TU (heavy CUTLASS sm100 templates). The candidate dispatch
// calls fp8_scaled_mm_swapab_smallm(...) for covered small-M shapes.
// CUTLASS include set replicated verbatim from the baseline fp8_gemm_kernel.cu so
// the 3.x Sm90* epilogue-fusion ops + visitors.hpp compile in the same context.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "fp8_swapab_smallm.h"

#if defined CUDA_VERSION && CUDA_VERSION >= 12080
using namespace cute;

template <typename CTAShape, typename ClusterShape>
struct SwapABGemmSm100 {
  using ElementType = cutlass::float_e4m3_t;
  using OutElementType = cutlass::bfloat16_t;
  using AccumElementType = float;
  using ElementComputeEpilogue = float;
  using TileShape = CTAShape;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  // ScaleA (col broadcast over the swapped-M axis = original N) <- scale_b[N].
  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0, TileShape, ElementComputeEpilogue, ElementComputeEpilogue,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;
  // ScaleB (row broadcast over the swapped-N axis = original M) <- scale_a[M].
  using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0, TileShape, ElementComputeEpilogue, ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;
  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementComputeEpilogue, ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, OutElementType, ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;

  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementType>::value;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementType>::value;
  using ElementC = void;
  using LayoutC = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;
  using LayoutD = cutlass::layout::ColumnMajor;  // out^T[N,M] column-major == row-major out[M,N]
  static constexpr int AlignmentD = AlignmentC;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, AccumElementType, ElementComputeEpilogue,
      ElementC, LayoutC, AlignmentC, OutElementType, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto, EVTCompute>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementType, LayoutA, AlignmentA,
      ElementType, LayoutB, AlignmentB, AccumElementType, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename G>
static void run_swapab(
    torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& scale_a, const torch::Tensor& scale_b) {
  using Gemm = typename G::Gemm;
  using GK = typename Gemm::GemmKernel;
  using StrideA = typename GK::StrideA;
  using StrideB = typename GK::StrideB;
  using StrideC = typename GK::StrideC;
  using StrideD = StrideC;

  const int M = a.size(0), N = b.size(1), K = a.size(1);  // original dims
  auto* ptr_a = reinterpret_cast<typename G::ElementType const*>(b.data_ptr());  // Bphys[N,K]
  auto* ptr_b = reinterpret_cast<typename G::ElementType const*>(a.data_ptr());  // A^T[K,M]

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(N, K, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(M, K, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, 1));
  StrideD stride_d = stride_c;

  typename GK::MainloopArguments mainloop_args{ptr_a, stride_a, ptr_b, stride_b};
  typename GK::ProblemShape prob_shape{N, M, K, 1};  // {m'=N, n'=M, k'=K, L}
  cutlass::KernelHardwareInfo hw_info;
  typename GK::TileSchedulerArguments scheduler{};

  auto* ptr_d = static_cast<typename G::OutElementType*>(out.data_ptr());
  // ScaleA <- scale_b, ScaleB <- scale_a (the swap). EVT: ScaleA*(ScaleB*acc).
  typename G::EVTCompute0::Arguments evt0_args{
      typename G::ScaleB::Arguments{reinterpret_cast<float const*>(scale_a.data_ptr())}, {}, {}};
  typename G::EVTCompute::Arguments epi_evt{
      typename G::ScaleA::Arguments{reinterpret_cast<float const*>(scale_b.data_ptr())}, evt0_args, {}};
  typename GK::EpilogueArguments epilogue_args{epi_evt, ptr_d, stride_c, ptr_d, stride_d};

  typename GK::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args, epilogue_args, hw_info, scheduler};

  Gemm gemm_op;
  size_t ws = gemm_op.get_workspace_size(args);
  auto ws_opts = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(static_cast<long>(ws), ws_opts);
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  TORCH_CHECK(gemm_op.can_implement(args) == cutlass::Status::kSuccess,
              "swap-AB can_implement failed");
  TORCH_CHECK(gemm_op.run(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess,
              "swap-AB run failed");
}

void fp8_scaled_mm_swapab_smallm(
    torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& scale_a, const torch::Tensor& scale_b) {
  const at::cuda::OptionalCUDAGuard guard(device_of(a));
  const int M = a.size(0);
  if (M <= 32) {
    using G = SwapABGemmSm100<Shape<_128, _32, _128>, Shape<_4, _1, _1>>;
    run_swapab<G>(out, a, b, scale_a, scale_b);
  } else {
    using G = SwapABGemmSm100<Shape<_128, _64, _256>, Shape<_4, _1, _1>>;
    run_swapab<G>(out, a, b, scale_a, scale_b);
  }
}
#else
void fp8_scaled_mm_swapab_smallm(torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                                 const torch::Tensor&, const torch::Tensor&) {
  TORCH_CHECK(false, "swap-AB requires CUDA >= 12.8");
}
#endif
