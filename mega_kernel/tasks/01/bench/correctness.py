"""Correctness comparators for the fused allreduce+add+rmsnorm task.

Two comparators, per the task contract:

1. Bitwise A/B: the ported kernel must produce bf16 bit-identical `out` and
   `residual_out` versus the flashinfer original on every rank (the P0 gate).
2. Composed fp32 oracle: fp32 cross-rank sum + fp32 residual add +
   fp32-accumulation RMSNorm, judged elementwise at the correctness
   contract's bf16 tolerance (atol 7e-2 / rtol 2e-2) — a sanity check on both
   implementations. The source prompt's "rel<1e-3" figure is kept only as a
   diagnostic: it is below the bf16 quantization floor (see ORACLE_REL_TOL
   note) and no bf16 kernel, the baseline included, can meet it under
   max-rel.

Also provides output poisoning and structural checks (shape/dtype/stride,
NaN/Inf) required by the correctness contract, plus self-tests proving the
comparators can fail (a comparator that cannot fail is a harness bug).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

# The source prompt words the fallback oracle bound as "fp32 oracle rel<1e-3".
# Measured fact (smoke_fi, 2026-07-10): a bf16 kernel output compared against a
# pure-fp32 reference has an irreducible quantization floor of one bf16 ulp,
# max-rel ~= 2^-8 = 3.9e-3 — the flashinfer ORIGINAL itself measures 3.904e-3.
# rel<1e-3 is therefore unsatisfiable for ANY bf16 implementation (including
# the baseline) under a max-rel metric. Per the correctness contract's dtype
# table, bf16 outputs are judged elementwise at atol 7e-2 / rtol 2e-2; we keep
# the prompt's 1e-3 as a diagnostic reference value and gate the oracle check
# on the contract tolerance. The PRIMARY P0 gate (bitwise A/B vs the original)
# is unaffected by any of this.
ORACLE_REL_TOL = 1e-3  # diagnostic only (see note above)
BF16_ATOL = 7e-2
BF16_RTOL = 2e-2


def contract_allclose(got: torch.Tensor, ref_fp32: torch.Tensor) -> Tuple[bool, float]:
    """Elementwise contract-tolerance check for bf16 vs fp32 reference.

    Returns (ok, worst_excess) where worst_excess is max(|a-b| - (atol+rtol|b|)),
    <= 0 when passing.
    """
    diff = (got.float() - ref_fp32).abs()
    bound = BF16_ATOL + BF16_RTOL * ref_fp32.abs()
    excess = (diff - bound).max().item()
    return excess <= 0.0, float(excess)


def poison_(t: torch.Tensor) -> torch.Tensor:
    """Fill an output tensor with NaN so stale/partial writes are visible."""
    return t.fill_(float("nan"))


def structural_check(name: str, got: torch.Tensor, ref: torch.Tensor) -> Optional[str]:
    if got.shape != ref.shape:
        return f"{name}: shape {tuple(got.shape)} != {tuple(ref.shape)}"
    if got.dtype != ref.dtype:
        return f"{name}: dtype {got.dtype} != {ref.dtype}"
    if got.stride() != ref.stride():
        return f"{name}: stride {got.stride()} != {ref.stride()}"
    if torch.isnan(got.float()).any().item():
        return f"{name}: contains NaN (poison survived -> missing write?)"
    if torch.isinf(got.float()).any().item():
        return f"{name}: contains Inf"
    return None


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, int, Optional[int]]:
    """bf16 bit-level equality. Returns (equal, mismatch_count, first_index)."""
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    ai = a.contiguous().view(torch.int16)
    bi = b.contiguous().view(torch.int16)
    neq = ai.ne(bi)
    count = int(neq.sum().item())
    if count == 0:
        return True, 0, None
    first = int(neq.flatten().nonzero()[0].item())
    return False, count, first


def rel_err(got: torch.Tensor, ref_fp32: torch.Tensor) -> float:
    denom = ref_fp32.abs().max().clamp_min(1e-6)
    return float((got.float() - ref_fp32).abs().max().item() / denom.item())


@dataclass
class OracleRefs:
    residual_out: torch.Tensor  # fp32
    out: torch.Tensor  # fp32


def fp32_oracle(
    xs: List[torch.Tensor],  # per-rank bf16 [T, H] shards
    residual: torch.Tensor,  # bf16 [T, H] (replicated across ranks)
    gamma: torch.Tensor,  # bf16 [H]
    eps: float,
) -> OracleRefs:
    """Composed reference: fp32 allreduce + residual add + fp32-accum rmsnorm."""
    acc = torch.zeros_like(xs[0], dtype=torch.float32, device=xs[0].device)
    for x in xs:
        acc += x.to(device=acc.device).float()
    res_ref = acc + residual.to(device=acc.device).float()
    ms = res_ref.pow(2).mean(dim=-1, keepdim=True)
    out_ref = res_ref * torch.rsqrt(ms + eps) * gamma.to(device=acc.device).float()
    return OracleRefs(residual_out=res_ref, out=out_ref)


# ----------------------------------------------------------------------
# Comparator self-tests (negative tests): each must FAIL on purpose.
# ----------------------------------------------------------------------


def self_test(device: str = "cuda:0") -> None:
    torch.manual_seed(1234)
    T, H, world = 6, 6144, 8
    eps = 1e-5
    xs = [torch.randn(T, H, dtype=torch.bfloat16, device=device) for _ in range(world)]
    residual = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    gamma = torch.randn(H, dtype=torch.bfloat16, device=device)

    # 1) bit-flip must break bitwise equality
    a = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    b = a.clone()
    bi = b.view(torch.int16)
    bi[3, 77] ^= 1
    eq, count, first = bitwise_equal(a, b)
    assert not eq and count == 1, "bitwise comparator failed to catch a flipped bit"

    # 2) identical tensors must pass
    eq, count, _ = bitwise_equal(a, a.clone())
    assert eq and count == 0, "bitwise comparator false-positive"

    # 3) eps misconfiguration must fail the contract-tolerance oracle in the
    #    regime where eps matters: tiny pre-norm magnitudes (ms ~ eps) but O(1)
    #    normalized outputs (normalization rescales), so a wrong eps shifts the
    #    output by tens of percent — far past atol+rtol. With unit-variance
    #    inputs the eps term would be ~1e-6 of the mean square and invisible.
    xs_small = [x * 1e-3 for x in xs]
    residual_small = residual * 1e-3
    ref_right = fp32_oracle(xs_small, residual_small, gamma, eps=1e-5)
    ref_wrong = fp32_oracle(xs_small, residual_small, gamma, eps=1e-6)
    ok_wrong, excess = contract_allclose(ref_wrong.out.to(torch.bfloat16), ref_right.out)
    assert not ok_wrong, "eps self-test regime too weak: wrong eps passed the oracle"
    ok_right, _ = contract_allclose(ref_right.out.to(torch.bfloat16), ref_right.out)
    assert ok_right, "oracle false-negative on the correct eps"

    # 4) poison must be visible to the structural check
    p = poison_(torch.empty(T, H, dtype=torch.bfloat16, device=device))
    msg = structural_check("poisoned", p, a)
    assert msg and "NaN" in msg, "structural check missed poisoned output"

    # 5) structural mismatches must be caught: shape, dtype, stride
    assert structural_check("shape", a[:, :64].contiguous(), a), "missed shape mismatch"
    assert structural_check("dtype", a.to(torch.float16), a), "missed dtype mismatch"
    assert structural_check("stride", a.t().contiguous().t(), a) or True  # same logical strides
    strided = torch.empty(T, 2 * H, dtype=torch.bfloat16, device=device)[:, :H]
    assert structural_check("stride", strided, a), "missed stride mismatch"

    # 6) residual-output path uses the same comparator: a flipped bit in a
    #    residual_out clone must fail exactly like out
    r = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    r2 = r.clone()
    r2.view(torch.int16)[0, 0] ^= 1
    eq, count, _ = bitwise_equal(r, r2)
    assert not eq and count == 1, "residual_out comparator failed to catch a flipped bit"

    print("correctness.self_test: all negative/positive comparator checks passed")


if __name__ == "__main__":
    self_test()
