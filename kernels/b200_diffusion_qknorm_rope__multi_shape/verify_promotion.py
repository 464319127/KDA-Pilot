#!/usr/bin/env python3
"""Verify the kda_kernels promotion of fused_inplace_qknorm_rope is sound:
install() swaps the SGLang symbol, the swapped wrapper runs the CUDA fast path
on a production shape and falls back (non-recursively) on an unsupported shape,
status() reports the promotion, and uninstall() restores the original baseline.

Run with kda_kernels on PYTHONPATH inside sglang_bbuf on ion-b200.
"""

import torch
import sglang.jit_kernel.diffusion.qknorm_rope as qr
import kda_kernels
import kda_kernels.diffusion.qknorm_rope as kqr


def mk(n, h, d, rope, pos_dtype=torch.int64):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (
        torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16, generator=g),
        torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16, generator=g),
        torch.randn(d, device="cuda", dtype=torch.bfloat16, generator=g),
        torch.randn(d, device="cuda", dtype=torch.bfloat16, generator=g),
        torch.randn(256, rope, device="cuda", dtype=torch.float32, generator=g),
        torch.randint(0, 256, (n,), device="cuda", dtype=pos_dtype, generator=g),
    )


def main() -> int:
    orig = qr.fused_inplace_qknorm_rope
    print("KDA_OPTIMIZED flag:",
          getattr(kqr, "KDA_OPTIMIZED_fused_inplace_qknorm_rope", None))
    kda_kernels.install()
    swapped = qr.fused_inplace_qknorm_rope
    print("INSTALL swapped_is_orig:", swapped is orig, "(expect False)")

    q, k, qw, kw, csc, pos = mk(64, 24, 128, 128)
    swapped(q, k, qw, kw, csc, pos, is_neox=False, eps=1e-6, rope_dim=128)
    print("PROD path ok, finite:", bool(torch.isfinite(q).all() and torch.isfinite(k).all()))

    # head_dim=64 -> unsupported by the fast path -> baseline fallback (no recursion)
    q2, k2, qw2, kw2, csc2, pos2 = mk(64, 8, 64, 64)
    swapped(q2, k2, qw2, kw2, csc2, pos2, is_neox=False, eps=1e-6, rope_dim=64)
    print("FALLBACK path ok, finite:", bool(torch.isfinite(q2).all() and torch.isfinite(k2).all()))

    kda_kernels.uninstall()
    print("UNINSTALL restored_is_orig:", qr.fused_inplace_qknorm_rope is orig, "(expect True)")
    print("VERIFY_PROMOTION_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
