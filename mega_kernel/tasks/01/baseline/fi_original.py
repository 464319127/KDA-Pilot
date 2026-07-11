"""Baseline runner: drives the flashinfer ORIGINAL MNNVL fused allreduce kernel.

The kernel executed here is the binary the production serving stack actually
loads: flashinfer's loader prefers the PRE-COMPILED AOT module from the
`flashinfer-jit-cache 0.6.12+cu130` wheel (a fast-math build — see
docs/results.md "verbatim source is not enough"); it does NOT JIT-compile
locally on this box. The wheel was built from the same source revision as
the header copied verbatim at `baseline/trtllm_mnnvl_allreduce.cuh` (sha256
ab6560f2..97aa; kernel symbol signatures verified identical). This module
never imports the live serving checkout and never imports the reference
copies under `baseline/upstream_ref/`.

Entry ABI (shared with the candidate, see bench/adapter.py):
    launch(input, output, residual_in, residual_out, gamma, ws_rank, epsilon,
           launch_with_pdl)
where `ws_rank` carries this rank's workspace pointers/flags and the call runs
under torch.cuda.device(ws_rank.device_index) on the current stream, with all
output tensors preallocated by the caller.
"""

from __future__ import annotations

import functools


@functools.cache
def _module():
    # Resolves to the precompiled flashinfer-jit-cache AOT module on this box
    # (fast-math build; no local JIT — see module docstring).
    from flashinfer.comm.trtllm_mnnvl_ar import get_trtllm_mnnvl_comm_module

    return get_trtllm_mnnvl_comm_module()


def name() -> str:
    return "flashinfer-original-0.6.12"


def launch(
    input,
    output,
    residual_in,
    residual_out,
    gamma,
    ws_rank,
    epsilon: float,
    launch_with_pdl: bool,
) -> None:
    """One fused allreduce+add+rmsnorm launch for a single rank (oneshot path).

    Caller guarantees: current device == ws_rank.device_index, tensors live on
    that device, outputs preallocated, world size and rank taken from ws_rank.
    """
    _module().trtllm_mnnvl_allreduce_fusion(
        input,
        ws_rank.mc_ptr,
        ws_rank.uc_ptrs_dev,
        ws_rank.uc_ptr_local,
        ws_rank.buffer_flags,
        ws_rank.world_size,
        ws_rank.rank,
        True,  # rmsnorm_fusion (frozen task semantics)
        launch_with_pdl,
        True,  # use_oneshot (the bs=1 production path under test)
        output,
        residual_out,
        residual_in,
        gamma,
        float(epsilon),
        0.0,  # weight_bias: standard RMSNorm (GLM-5.2)
    )
