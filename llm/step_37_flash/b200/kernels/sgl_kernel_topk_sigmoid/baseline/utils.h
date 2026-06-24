// Minimal local shim for the symbols that the vendored upstream kernel
// (topk_sigmoid_baseline.cu) references from SGLang's sgl-kernel/include/utils.h.
//
// The vendored kernel only uses WARP_SIZE and SGLANG_SHFL_XOR_SYNC_WIDTH, and
// both are only reached on the power-of-two-expert fast path (NOT used for the
// captured 288-expert workload, which takes the workspace path). Definitions are
// copied verbatim (CUDA branch) from upstream sgl-kernel/include/utils.h so the
// baseline behaves identically. We provide a minimal shim instead of vendoring
// the full upstream header to keep the local baseline self-contained and free of
// the broader sgl-kernel header cascade. Provenance: docs/baseline_source.md.
#pragma once

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef SGLANG_SHFL_XOR_SYNC
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) __shfl_xor_sync((mask), (var), (lane_mask))
#endif

#ifndef SGLANG_SHFL_XOR_SYNC_WIDTH
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) \
  __shfl_xor_sync((mask), (var), (lane_mask), (width))
#endif
