#!/usr/bin/env python3
"""Single-launch profiling entrypoint for Nsight Compute.

Profiles exactly one launch of the chosen implementation on one captured shape,
bracketed by cudaProfilerStart/Stop so `ncu --profile-from-start off` captures
only that launch (warmup and input resets are excluded). The candidate CUDA
extension is built with -lineinfo so SASS maps back to source.

Env:
  PROFILE_CASE  capture case name (default qwen__4096)
  PROFILE_IMPL  'candidate' | 'baseline' (default candidate)
"""

import importlib.util
import os
import sys

import torch

KDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(KDIR, "src"))


def _load_correctness():
    spec = importlib.util.spec_from_file_location(
        "kdac", os.path.join(KDIR, "tests", "test_correctness.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    m = _load_correctness()
    import wrapper

    wrapper.build()
    case_name = os.environ.get("PROFILE_CASE", "qwen__4096")
    impl = os.environ.get("PROFILE_IMPL", "candidate")
    case = next(c for c in m.make_cases() if c["name"] == case_name)
    inp = m.build_inputs(case)
    qw, kw = inp["q_weight"], inp["k_weight"]
    csc, pos = inp["cos_sin_cache"], inp["positions"]
    is_neox, eps, rope_dim = inp["is_neox"], inp["eps"], inp["rope_dim"]
    q0, k0 = inp["q"].clone(), inp["k"].clone()
    qb, kb = q0.clone(), k0.clone()

    if impl == "candidate":
        def fn():
            wrapper.fused_inplace_qknorm_rope(
                qb, kb, qw, kw, csc, pos, is_neox=is_neox, eps=eps, rope_dim=rope_dim)
    else:
        from sglang.jit_kernel.diffusion.qknorm_rope import fused_inplace_qknorm_rope as bfn

        def fn():
            bfn(qb, kb, qw, kw, csc, pos, is_neox=is_neox, eps=eps, rope_dim=rope_dim)

    for _ in range(30):
        qb.copy_(q0); kb.copy_(k0); fn()
    torch.cuda.synchronize()
    qb.copy_(q0); kb.copy_(k0)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"profiled impl={impl} case={case_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
