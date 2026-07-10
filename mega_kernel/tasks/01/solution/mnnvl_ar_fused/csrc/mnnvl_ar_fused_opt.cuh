// bs=1 specialization of the verbatim oneshot fused AR kernel
// (mnnvl_ar_fused.cuh): NumTokens and TokenDim become template constants so
// index arithmetic folds and loops unroll fully, while every VALUE-BEARING
// operation keeps the exact arithmetic, order, and launch geometry of the
// generic kernel — required because the e2e promote gate demands greedy
// output byte-identical to the stock path:
//   - the per-thread world-sum and the norm reduction tree are untouched;
//   - the launch geometry comes from the same adjustGridConfig, so the
//     cross-warp/cluster fp32 sum order is identical;
//   - the "fullSum / tokenDim" division uses the RUNTIME tokenDim kernel
//     argument (not the constant) so the compiler cannot fold the division
//     into a reciprocal multiply and change the quotient bits.
// Uncovered shape/parameter combinations fall back to the generic verbatim
// dispatch, which itself matches the deployed flashinfer binary.
#pragma once

#include "mnnvl_ar_fused.cuh"

namespace flashinfer {
namespace trtllm_mnnvl_allreduce {

template <uint8_t WorldSize, typename T, int NumTokens, int TokenDim,
          typename PackedType = float4>
__global__ void __launch_bounds__(1024)
    oneshotArFusedNormConstKernel(T* outputPtr, T* prenormedPtr, T const* shardPtr,
                                  T const* residualInPtr, T const* gammaPtr, T** inputPtrs,
                                  T* mcastPtr, int const numTokens, int const tokenDim,
                                  float epsilon, float weightBias, int const rank,
                                  uint32_t* bufferFlags) {
  constexpr int kELTS_PER_THREAD = sizeof(PackedType) / sizeof(T);
  constexpr int kLAMPORT_ELTS_PER_PACKED = sizeof(PackedType) / sizeof(float);
  constexpr uint32_t kELT_SIZE = sizeof(T);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  int packedIdx = cluster.thread_rank();
  int token = blockIdx.x;
  int threadOffset = token * TokenDim + packedIdx * kELTS_PER_THREAD;

  cudaGridDependencySynchronize();
#else
  int packedIdx = blockIdx.y * blockDim.x + threadIdx.x;
  int token = blockIdx.x;
  int threadOffset = token * TokenDim + packedIdx * kELTS_PER_THREAD;
#endif

  LamportFlags<PackedType> flag(bufferFlags, 1);
  T* stagePtrMcast = reinterpret_cast<T*>(flag.getCurLamportBuf(mcastPtr, 0));
  T* stagePtrLocal = reinterpret_cast<T*>(flag.getCurLamportBuf(inputPtrs[rank], 0));

  if (packedIdx * kELTS_PER_THREAD >= TokenDim) {
    flag.ctaArrive();
    flag.clearDirtyLamportBuf(inputPtrs[rank], -1);
    return;
  }

  PackedVec<PackedType, T> val;
  val.packed = loadPacked<PackedType>(&shardPtr[threadOffset]);
#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    if (isNegZero(val.elements[i])) val.elements[i] = fromFloat<T>(0.f);
  }

  reinterpret_cast<PackedType*>(
      &stagePtrMcast[token * TokenDim * WorldSize + rank * TokenDim])[packedIdx] = val.packed;

  flag.ctaArrive();
  flag.clearDirtyLamportBuf(inputPtrs[rank], -1);

  // Prefetch the norm operands BEFORE the Lamport wait so their global-load
  // latency hides behind the peer-arrival spin. Same addresses, same values,
  // same later use order — value-identical to the generic kernel; only the
  // load ISSUE time moves.
  PackedVec<PackedType, T> residualIn;
  residualIn.packed = *reinterpret_cast<PackedType const*>(&residualInPtr[threadOffset]);
  PackedVec<PackedType, T> gamma;
  gamma.packed = *reinterpret_cast<PackedType const*>(&gammaPtr[packedIdx * kELTS_PER_THREAD]);

  PackedVec<PackedType, float> valuesLamport[WorldSize];
  while (1) {
    bool valid = true;
#pragma unroll
    for (int r = 0; r < WorldSize; r++) {
      valuesLamport[r].packed = loadPackedVolatile<PackedType>(
          &stagePtrLocal[token * TokenDim * WorldSize + r * TokenDim +
                         packedIdx * kELTS_PER_THREAD]);

#pragma unroll
      for (int i = 0; i < kLAMPORT_ELTS_PER_PACKED; i++) {
        valid &= !isNegZero(valuesLamport[r].elements[i]);
      }
    }
    if (valid) {
      break;
    }
  }

  auto values = reinterpret_cast<PackedVec<PackedType, T>*>(valuesLamport);
  float accum[kELTS_PER_THREAD];
  PackedVec<PackedType, T> packedAccum;

#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    accum[i] = toFloat<T>(values[0].elements[i]);
  }

#pragma unroll
  for (int r = 1; r < WorldSize; r++) {
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      accum[i] += toFloat<T>(values[r].elements[i]);
    }
  }

#pragma unroll
  for (int i = 0; i < kELTS_PER_THREAD; i++) {
    packedAccum.elements[i] = fromFloat<T>(accum[i]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  {
    packedAccum += residualIn;
    *reinterpret_cast<PackedType*>(&prenormedPtr[threadOffset]) = packedAccum.packed;

    float threadSum = 0.F;
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      threadSum += toFloat<T>(packedAccum.elements[i] * packedAccum.elements[i]);
    }
    float blockSum = blockReduceSum<float, true>(threadSum);

    __shared__ float sharedVal[8];
    float fullSum = blockSum;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int const numBlocks = cluster.num_blocks();
    if (numBlocks > 1) {
      fullSum = 0.F;
      int const blockRank = cluster.block_rank();
      if (threadIdx.x < numBlocks) {
        cluster.map_shared_rank(&sharedVal[0], threadIdx.x)[blockRank] = blockSum;
      }
      cluster.barrier_wait(cluster.barrier_arrive());
      for (int i = 0; i < numBlocks; ++i) {
        fullSum += sharedVal[i];
      }
    }
#endif
    // RUNTIME tokenDim on purpose: keeps the division emission (and thus the
    // quotient bits) identical to the generic kernel / deployed baseline.
    float rcpRms = rsqrtf(fullSum / tokenDim + epsilon);
#pragma unroll
    for (int i = 0; i < kELTS_PER_THREAD; i++) {
      packedAccum.elements[i] = fromFloat<T>(toFloat<T>(packedAccum.elements[i]) * rcpRms *
                                             (weightBias + toFloat<T>(gamma.elements[i])));
    }
  }
  reinterpret_cast<PackedType*>(&outputPtr[threadOffset])[0] = packedAccum.packed;
  flag.waitAndUpdate(
      {static_cast<uint32_t>(NumTokens * TokenDim * WorldSize * kELT_SIZE), 0, 0, 0});
}

// Specialized dispatch: constant instantiations for the frozen bs=1 shapes,
// identical launch geometry via the same adjustGridConfig; anything not
// covered falls back to the generic verbatim dispatch (correctness is never
// lost; dispatch cost is a handful of scalar compares on the host).
template <typename T>
cudaError_t oneshotArFusedConstDispatch(AllReduceFusionParams const& params) {
  bool const specialShape =
      params.rmsNormFusion && params.nRanks == 8 && params.tokenDim == 6144 &&
      (params.numTokens == 6 || params.numTokens == 1) &&
      std::is_same_v<T, __nv_bfloat16>;
  if (!specialShape) {
    return oneshotAllreduceFusionDispatch<T>(params);
  }

  int const numTokens = params.numTokens;
  int const tokenDim = params.tokenDim;
  int const eltsPerThread = sizeof(float4) / sizeof(T);
  static const int kSMVersionMajor = GetCudaComputeCapability().first;

  auto [blockSize, clusterSize, loadsPerThread] =
      adjustGridConfig(numTokens, tokenDim, eltsPerThread, kSMVersionMajor);
  dim3 grid(numTokens, clusterSize, 1);

  FLASHINFER_CHECK(blockSize <= 1024 && loadsPerThread == 1,
                   "Hidden Dimension exceeds the maximum supported hidden dimension");

  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = params.launchWithPdl ? 1 : 0;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = 1;
  attrs[1].val.clusterDim.y = clusterSize;
  attrs[1].val.clusterDim.z = 1;

  cudaLaunchConfig_t config{
      .gridDim = grid,
      .blockDim = static_cast<unsigned int>(blockSize),
      .dynamicSmemBytes = 0,
      .stream = params.stream,
      .attrs = attrs,
      .numAttrs = kSMVersionMajor >= 9 ? 2 : 1,
  };

  T** ucPtrs = reinterpret_cast<T**>(params.bufferPtrsDev);
  T* mcPtr = reinterpret_cast<T*>(params.multicastPtr);
  T* output = reinterpret_cast<T*>(params.output);
  T* residualOut = reinterpret_cast<T*>(params.residualOut);
  T const* input = reinterpret_cast<T const*>(params.input);
  T const* residualIn = reinterpret_cast<T const*>(params.residualIn);
  T const* gamma = reinterpret_cast<T const*>(params.gamma);

#define LAUNCH_CONST_KERNEL(NTOK)                                                            \
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(                                                   \
      &config, &oneshotArFusedNormConstKernel<8, T, NTOK, 6144>, output, residualOut, input, \
      residualIn, gamma, ucPtrs, mcPtr, numTokens, tokenDim,                                 \
      static_cast<float>(params.epsilon), params.weightBias, params.rank,                    \
      params.bufferFlags));
  if (numTokens == 6) {
    LAUNCH_CONST_KERNEL(6);
  } else {
    LAUNCH_CONST_KERNEL(1);
  }
#undef LAUNCH_CONST_KERNEL
  return cudaSuccess;
}

}  // namespace trtllm_mnnvl_allreduce
}  // namespace flashinfer
