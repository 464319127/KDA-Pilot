"""Experimental runner: block-arrival flag-granularity variant (route eval).

Fence/flag-granularity P1 route closure — measurement only, never a serving
path. Same module .so as jit_port; distinct symbol.
"""

from __future__ import annotations

from solution import jit_port


def name() -> str:
    return "jit-port-opt-ba(block-arrival)"


def launch(input, output, residual_in, residual_out, gamma, ws_rank,
           epsilon: float, launch_with_pdl: bool) -> None:
    jit_port._module().mnnvl_ar_fused_opt_ba(
        input, ws_rank.mc_ptr, ws_rank.uc_ptrs_dev, ws_rank.uc_ptr_local,
        ws_rank.buffer_flags, ws_rank.world_size, ws_rank.rank,
        True, launch_with_pdl, True,
        output, residual_out, residual_in, gamma, float(epsilon), 0.0,
    )
