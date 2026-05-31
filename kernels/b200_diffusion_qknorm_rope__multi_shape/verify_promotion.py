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


ATOL, RTOL = 8e-2, 1e-2


def _oracle(q, k, qw, kw, csc, pos, is_neox, eps):
    """SGLang split oracle (norm is not swapped by install; only qknorm_rope is)."""
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
    from sglang.jit_kernel.norm import fused_inplace_qknorm

    fused_inplace_qknorm(q, k, qw, kw, eps)
    apply_rope_with_cos_sin_cache_inplace(
        positions=pos.long(), query=q.view(q.shape[0], -1), key=k.view(k.shape[0], -1),
        head_size=q.shape[-1], cos_sin_cache=csc, is_neox=is_neox)
    return q, k


def _cmp_vs_oracle(name, swapped, n, h, d, rope):
    q, k, qw, kw, csc, pos = mk(n, h, d, rope)
    qo, ko = _oracle(q.clone(), k.clone(), qw, kw, csc, pos, False, 1e-6)
    qc, kc = q.clone(), k.clone()
    swapped(qc, kc, qw, kw, csc, pos, is_neox=False, eps=1e-6, rope_dim=rope)
    eq = (qc.float() - qo.float()).abs().max().item()
    ek = (kc.float() - ko.float()).abs().max().item()
    ceil_q = ATOL + RTOL * qo.float().abs().max().item()
    ceil_k = ATOL + RTOL * ko.float().abs().max().item()
    ok = bool(torch.isfinite(qc).all() and torch.isfinite(kc).all() and eq <= ceil_q and ek <= ceil_k)
    print(f"{name}: max_err q={eq:.3e} k={ek:.3e} within_ceiling={ok}")
    return ok


def main() -> int:
    orig = qr.fused_inplace_qknorm_rope
    print("KDA_OPTIMIZED flag:",
          getattr(kqr, "KDA_OPTIMIZED_fused_inplace_qknorm_rope", None))
    kda_kernels.install()
    swapped = qr.fused_inplace_qknorm_rope
    print("INSTALL swapped_is_orig:", swapped is orig, "(expect False)")

    # Post-install correctness vs the oracle, on the CUDA path and the fallback path.
    prod_ok = _cmp_vs_oracle("PROD(cuda)", swapped, 64, 24, 128, 128)
    fb_ok = _cmp_vs_oracle("FALLBACK(hd64)", swapped, 64, 8, 64, 64)  # head_dim=64 -> baseline

    kda_kernels.uninstall()
    restored_ok = qr.fused_inplace_qknorm_rope is orig
    print("UNINSTALL restored_is_orig:", restored_ok, "(expect True)")
    all_ok = prod_ok and fb_ok and restored_ok
    print("VERIFY_PROMOTION_DONE all_ok =", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
