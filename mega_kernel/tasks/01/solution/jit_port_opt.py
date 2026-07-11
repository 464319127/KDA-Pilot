"""P1 candidate runner: the bs=1 constant-specialized kernel entry.

Same module .so as jit_port (mnnvl_ar_fused_opt symbol); frozen shapes route
to the constant instantiation, everything else falls back to the generic
verbatim dispatch inside the same call.
"""

from __future__ import annotations

from solution import jit_port


def name() -> str:
    return "jit-port-opt(bs1-const)"


def launch(input, output, residual_in, residual_out, gamma, ws_rank,
           epsilon: float, launch_with_pdl: bool) -> None:
    jit_port._module().mnnvl_ar_fused_opt(
        input,
        ws_rank.mc_ptr,
        ws_rank.uc_ptrs_dev,
        ws_rank.uc_ptr_local,
        ws_rank.buffer_flags,
        ws_rank.world_size,
        ws_rank.rank,
        True,
        launch_with_pdl,
        True,
        output,
        residual_out,
        residual_in,
        gamma,
        float(epsilon),
        0.0,
    )
