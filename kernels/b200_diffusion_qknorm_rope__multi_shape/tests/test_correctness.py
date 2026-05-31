"""Correctness harness for ``b200_diffusion_qknorm_rope__multi_shape``.

Fused in-place QK-Norm + RoPE on (Q, K). The optimized candidate is compared
against the SGLang split oracle (``fused_inplace_qknorm`` + FlashInfer RoPE),
exactly as the upstream SGLang test does, plus a pure-fp32 reference used for a
dynamic, BF16-quantization-aware tolerance.

Skipped unless ``KDA_RUN_CORRECTNESS=1`` is set (the SGLang baseline / oracle
and FlashInfer live only in the remote ion-b200 environment).

Shared with ``benchmark.py``: ``make_cases()``, ``build_inputs()`` and the
``run_oracle`` / ``run_sglang_baseline`` / ``run_candidate`` callables.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Any, Callable

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - CUDA env owns the real run
    torch = None


KERNEL_SLUG = "b200_diffusion_qknorm_rope__multi_shape"
OP_TYPE = "qknorm_rope_inplace"
KERNEL_DIR = Path(__file__).resolve().parents[1]

# SGLang test contract (recovered): tolerance ceiling, RoPE base, cache length.
ATOL = 8e-2
RTOL = 1e-2
ROPE_BASE = 10000.0
MAX_SEQ_LEN = 131072
# Dynamic tolerance: candidate error vs fp32 must stay within TOL_MULT x the
# BF16-quantization noise of the oracle vs fp32, with a small absolute floor.
DYNAMIC_TOL_MULT = 4.0
DYNAMIC_TOL_FLOOR = 2e-3

pytestmark = pytest.mark.skipif(
    os.environ.get("KDA_RUN_CORRECTNESS") != "1",
    reason="Set KDA_RUN_CORRECTNESS=1 on an ion-b200 GPU to run correctness.",
)


# --------------------------------------------------------------------------- #
# Register module (optimized candidate)
# --------------------------------------------------------------------------- #
def _load_register_module():
    register_py = KERNEL_DIR / "src" / "register.py"
    spec = importlib.util.spec_from_file_location(
        f"kda_kernel_{KERNEL_SLUG}_register", register_py
    )
    assert spec is not None and spec.loader is not None, register_py
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
# The 10 captured production shapes (docs/captured_shapes_b200.jsonl). All are
# bf16, head_dim=128, rope_dim=128, is_neox=False; eps differs (zimage=1e-5).
_PRODUCTION = [
    ("joyai_edit__7904", 7904, 32, 1e-6),
    ("qwen__4096", 4096, 24, 1e-6),
    ("qwen__19", 19, 24, 1e-6),
    ("qwen__47", 47, 24, 1e-6),
    ("qwen_edit__8424", 8424, 24, 1e-6),
    ("qwen_edit__195", 195, 24, 1e-6),
    ("qwen_edit__189", 189, 24, 1e-6),
    ("zimage__4096", 4096, 30, 1e-5),
    ("zimage__32", 32, 30, 1e-5),
    ("zimage__4128", 4128, 30, 1e-5),
]

# Non-production tuples. ``expect_path`` records whether the fast path handles
# the tuple natively ("cuda") or must route to the SGLang baseline ("fallback").
# Either way the result must match the oracle.
#   name, tokens, heads, head_dim, rope_dim, is_neox, eps, pos_dtype, expect_path
_OTHER = [
    # int32 positions at the production signature ARE covered natively.
    ("supported__hd128_int32pos", 129, 24, 128, 128, False, 1e-6, "int32", "cuda"),
    # Genuinely outside the fast path -> baseline fallback.
    ("fallback__hd64_rope64", 512, 16, 64, 64, False, 1e-6, "int64", "fallback"),
    ("fallback__hd256_rope128_neox", 512, 16, 256, 128, True, 1e-6, "int64", "fallback"),
    ("fallback__hd128_rope64", 257, 8, 128, 64, False, 1e-6, "int64", "fallback"),
    ("fallback__hd128_neox", 256, 8, 128, 128, True, 1e-6, "int64", "fallback"),
]


def make_cases() -> list[dict[str, Any]]:
    """All configured correctness/benchmark cases."""
    cases: list[dict[str, Any]] = []
    for name, tokens, heads, eps in _PRODUCTION:
        cases.append(
            {
                "name": name,
                "bucket": "tiny" if tokens < 256 else "large",
                "kind": "production",
                "num_tokens": tokens,
                "num_heads": heads,
                "head_dim": 128,
                "rope_dim": 128,
                "is_neox": False,
                "eps": eps,
                "pos_dtype": "int64",
                "expect_path": "cuda",
                "warmup": 30,
                "iters": 200,
            }
        )
    for name, tokens, heads, hd, rd, neox, eps, pos, expect in _OTHER:
        cases.append(
            {
                "name": name,
                "bucket": "other",
                "kind": "other",
                "num_tokens": tokens,
                "num_heads": heads,
                "head_dim": hd,
                "rope_dim": rd,
                "is_neox": neox,
                "eps": eps,
                "pos_dtype": pos,
                "expect_path": expect,
                "warmup": 10,
                "iters": 50,
            }
        )
    return cases


# --------------------------------------------------------------------------- #
# Inputs (deterministic, matches the SGLang test setup)
# --------------------------------------------------------------------------- #
def _create_cos_sin_cache(rope_dim: int, max_position: int = MAX_SEQ_LEN) -> "torch.Tensor":
    inv_freq = 1.0 / (
        ROPE_BASE
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device="cuda") / rope_dim)
    )
    t = torch.arange(max_position, dtype=torch.float32, device="cuda")
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def build_inputs(case: dict[str, Any], *, positions_mode: str = "random") -> dict[str, Any]:
    """Deterministic per-case inputs. ``positions_mode`` is 'random' (default,
    matches the SGLang test) or 'arange' (production-like contiguous stress)."""
    assert torch is not None
    n = case["num_tokens"]
    h = case["num_heads"]
    d = case["head_dim"]
    rope_dim = case["rope_dim"]
    seed = (
        n * 1_000_003 + h * 8191 + d * 127 + rope_dim * 17 + int(case["is_neox"]) * 3
    )
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    pos_dtype = torch.int32 if case["pos_dtype"] == "int32" else torch.int64
    if positions_mode == "arange":
        positions = (torch.arange(n, device="cuda") % MAX_SEQ_LEN).to(pos_dtype)
    else:
        positions = torch.randint(
            0, MAX_SEQ_LEN, (n,), device="cuda", dtype=pos_dtype, generator=g
        )
    return {
        "q": torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16, generator=g),
        "k": torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16, generator=g),
        "q_weight": torch.randn(d, device="cuda", dtype=torch.bfloat16, generator=g),
        "k_weight": torch.randn(d, device="cuda", dtype=torch.bfloat16, generator=g),
        "cos_sin_cache": _create_cos_sin_cache(rope_dim),
        "positions": positions,
        "is_neox": bool(case["is_neox"]),
        "eps": float(case["eps"]),
        "head_dim": d,
        "rope_dim": rope_dim,
    }


# --------------------------------------------------------------------------- #
# Implementations (operate in place on the passed q, k)
# --------------------------------------------------------------------------- #
def run_oracle(q, k, inp) -> tuple:
    """SGLang split oracle: fused_inplace_qknorm(eps) + FlashInfer RoPE."""
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    from sglang.jit_kernel.norm import fused_inplace_qknorm

    fused_inplace_qknorm(q, k, inp["q_weight"], inp["k_weight"], inp["eps"])
    apply_rope_with_cos_sin_cache_inplace(
        positions=inp["positions"].long(),
        query=q.view(q.shape[0], -1),
        key=k.view(k.shape[0], -1),
        head_size=q.shape[-1],
        cos_sin_cache=inp["cos_sin_cache"],
        is_neox=inp["is_neox"],
    )
    return q, k


def run_sglang_baseline(q, k, inp) -> tuple:
    """SGLang fused baseline (the kernel under optimization / bench baseline)."""
    from sglang.jit_kernel.diffusion.qknorm_rope import fused_inplace_qknorm_rope

    fused_inplace_qknorm_rope(
        q,
        k,
        inp["q_weight"],
        inp["k_weight"],
        inp["cos_sin_cache"],
        inp["positions"],
        is_neox=inp["is_neox"],
        eps=inp["eps"],
        rope_dim=inp["rope_dim"],
    )
    return q, k


def run_candidate(q, k, inp) -> tuple:
    """Optimized candidate via src/register.py optimized_wrapper."""
    module = _load_register_module()
    module.optimized_wrapper(
        q,
        k,
        inp["q_weight"],
        inp["k_weight"],
        inp["cos_sin_cache"],
        inp["positions"],
        is_neox=inp["is_neox"],
        eps=inp["eps"],
        rope_dim=inp["rope_dim"],
    )
    return q, k


def torch_reference_fp32(inp) -> tuple:
    """Pure-fp32 RMSNorm(full head_dim, eps) + RoPE reference for dynamic tol."""
    assert torch is not None
    eps = inp["eps"]
    rope_dim = inp["rope_dim"]
    is_neox = inp["is_neox"]
    cache = inp["cos_sin_cache"]
    pos = inp["positions"].long()
    qw = inp["q_weight"].float()
    kw = inp["k_weight"].float()

    def _norm(x):
        x32 = x.float()
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        return x32 * torch.rsqrt(var + eps)

    def _rope(x32, w):
        out = x32 * w  # [N, H, D]
        rotated = out[..., :rope_dim].clone()
        half = rope_dim // 2
        cos = cache[pos, :half].to(torch.float32)[:, None, :]  # [N,1,half]
        sin = cache[pos, half:rope_dim].to(torch.float32)[:, None, :]
        if is_neox:
            x1 = rotated[..., :half]
            x2 = rotated[..., half:]
            out[..., :rope_dim] = torch.cat(
                (x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1
            )
        else:
            x_even = rotated[..., 0::2]
            x_odd = rotated[..., 1::2]
            new_even = x_even * cos - x_odd * sin
            new_odd = x_odd * cos + x_even * sin
            inter = torch.stack((new_even, new_odd), dim=-1).flatten(start_dim=-2)
            out[..., :rope_dim] = inter
        return out

    qref = _rope(_norm(inp["q"]), qw)
    kref = _rope(_norm(inp["k"]), kw)
    return qref, kref


# --------------------------------------------------------------------------- #
# Assertions
# --------------------------------------------------------------------------- #
def _assert_finite(t, *, name: str) -> None:
    assert not torch.isnan(t).any(), f"{name} has NaN"
    assert not torch.isinf(t).any(), f"{name} has Inf"


def _max_abs(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def _check_against_oracle(cand, oracle, ref32, *, name: str) -> dict:
    """Hard ceiling vs oracle + dynamic fp32-anchored bound. Returns metrics."""
    _assert_finite(cand, name=f"{name}")
    # bf16 quantization noise of the oracle itself, relative to fp32.
    bf16_noise = _max_abs(oracle, ref32)
    cand_err = _max_abs(cand, ref32)
    dyn_bound = max(DYNAMIC_TOL_FLOOR, DYNAMIC_TOL_MULT * bf16_noise)
    # Hard ceiling vs the bf16 oracle (mirrors the upstream SGLang test).
    ceil_err = _max_abs(cand, oracle)
    metrics = {
        "bf16_noise": bf16_noise,
        "cand_err_vs_fp32": cand_err,
        "dyn_bound": dyn_bound,
        "cand_err_vs_oracle": ceil_err,
    }
    assert ceil_err <= ATOL + RTOL * oracle.float().abs().max().item(), (
        f"{name}: candidate vs oracle max abs err {ceil_err:.3e} exceeds ceiling "
        f"(ATOL={ATOL}, RTOL={RTOL}); metrics={metrics}"
    )
    assert cand_err <= dyn_bound, (
        f"{name}: candidate vs fp32 err {cand_err:.3e} exceeds dynamic bound "
        f"{dyn_bound:.3e} (bf16 noise {bf16_noise:.3e}); metrics={metrics}"
    )
    return metrics


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_register_metadata() -> None:
    module = _load_register_module()
    assert hasattr(module, "register")
    spec = module.register()
    assert spec["name"] == KERNEL_SLUG
    assert spec["op_type"] == OP_TYPE
    assert callable(spec["callable"])


def test_oracle_matches_sglang_fused() -> None:
    """Sanity (no candidate needed): the SGLang fused baseline matches the split
    oracle within the ceiling, and both stay within the fp32 dynamic bound.
    Validates the harness/oracle plumbing independently of our kernel."""
    for case in make_cases():
        if case["kind"] != "production":
            continue
        inp = build_inputs(case)
        ref32 = torch_reference_fp32(inp)
        q_o, k_o = run_oracle(inp["q"].clone(), inp["k"].clone(), inp)
        q_f, k_f = run_sglang_baseline(inp["q"].clone(), inp["k"].clone(), inp)
        _assert_finite(q_o, name=f"{case['name']}/oracle.q")
        _assert_finite(k_o, name=f"{case['name']}/oracle.k")
        _check_against_oracle(q_f, q_o, ref32[0], name=f"{case['name']}/sglang.q")
        _check_against_oracle(k_f, k_o, ref32[1], name=f"{case['name']}/sglang.k")


def _candidate_dispatch_path():
    """Path taken by the most recent candidate call ('cuda' | 'fallback')."""
    try:
        import wrapper  # src/ is on sys.path after _load_register_module()

        return wrapper.last_dispatch_path()
    except Exception:
        return None


def _run_candidate_case(case: dict[str, Any]) -> None:
    inp = build_inputs(case)
    ref32 = torch_reference_fp32(inp)
    q_o, k_o = run_oracle(inp["q"].clone(), inp["k"].clone(), inp)
    try:
        q_c, k_c = run_candidate(inp["q"].clone(), inp["k"].clone(), inp)
    except NotImplementedError:
        pytest.skip("optimized_wrapper not implemented yet (stub)")
    # AC-3/AC-4: confirm the intended path actually ran, not inferred.
    expected_path = case["expect_path"]
    actual_path = _candidate_dispatch_path()
    assert actual_path == expected_path, (
        f"{case['name']}: dispatch path {actual_path!r} != expected {expected_path!r}"
    )
    _check_against_oracle(q_c, q_o, ref32[0], name=f"{case['name']}/cand.q")
    _check_against_oracle(k_c, k_o, ref32[1], name=f"{case['name']}/cand.k")


@pytest.mark.parametrize(
    "case", [c for c in make_cases() if c["kind"] == "production"], ids=lambda c: c["name"]
)
def test_candidate_production(case: dict[str, Any]) -> None:
    _run_candidate_case(case)


@pytest.mark.parametrize(
    "case", [c for c in make_cases() if c["kind"] == "other"], ids=lambda c: c["name"]
)
def test_candidate_other(case: dict[str, Any]) -> None:
    """Non-production tuples: each takes its expected path and matches the oracle
    (int32 production signature natively; head_dim/rope/neox tail via fallback)."""
    _run_candidate_case(case)


def test_candidate_arange_positions() -> None:
    """Production-like contiguous positions stress (AC-2.1)."""
    case = make_cases()[0]  # joyai_edit__7904
    inp = build_inputs(case, positions_mode="arange")
    ref32 = torch_reference_fp32(inp)
    q_o, k_o = run_oracle(inp["q"].clone(), inp["k"].clone(), inp)
    try:
        q_c, k_c = run_candidate(inp["q"].clone(), inp["k"].clone(), inp)
    except NotImplementedError:
        pytest.skip("optimized_wrapper not implemented yet (stub)")
    assert _candidate_dispatch_path() == "cuda", "arange production case must take CUDA path"
    _check_against_oracle(q_c, q_o, ref32[0], name=f"{case['name']}/arange.q")
    _check_against_oracle(k_c, k_o, ref32[1], name=f"{case['name']}/arange.k")


def test_dispatch_gate() -> None:
    """AC-4: the fast-path gate is airtight — only true production tuples pass;
    malformed or foreign-device tuples are rejected (would route to fallback)."""
    _load_register_module()  # puts src/ on sys.path
    import wrapper

    def mk(d=128, rope=128, q_dtype=torch.bfloat16, pos_dtype=torch.int64,
           pos_contig=True, cache_dim=2, posdev="cuda"):
        n, h = 64, 8
        q = torch.randn(n, h, d, device="cuda", dtype=q_dtype)
        k = torch.randn(n, h, d, device="cuda", dtype=q_dtype)
        qw = torch.randn(d, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn(d, device="cuda", dtype=torch.bfloat16)
        csc = (torch.randn(256, rope, device="cuda", dtype=torch.float32)
               if cache_dim == 2 else torch.randn(256, 1, rope, device="cuda", dtype=torch.float32))
        pos = torch.zeros(n if pos_contig else 2 * n, device=posdev, dtype=pos_dtype)
        if not pos_contig:
            pos = pos[::2]
        return q, k, qw, kw, csc, pos

    def sup(is_neox=False, **kw):
        q, k, qw, kw_, csc, pos = mk(**kw)
        rope = csc.size(-1)
        return wrapper._supported(q, k, qw, kw_, csc, pos, is_neox, q.size(-1), rope)

    assert sup() is True                       # production signature
    assert sup(is_neox=True) is False          # neox -> fallback
    assert sup(q_dtype=torch.float16) is False  # fp16 -> fallback
    assert sup(d=64, rope=64) is False          # head_dim 64 -> fallback
    assert sup(pos_contig=False) is False       # non-contiguous positions
    assert sup(cache_dim=3) is False            # [N,1,128] cache rank
    assert sup(posdev="cpu") is False           # foreign-device positions


@pytest.mark.parametrize("case_name,scale", [("qwen__4096", 1e-3), ("zimage__4096", 1e-3)])
def test_eps_sensitivity(case_name: str, scale: float) -> None:
    """Tiny-magnitude inputs make eps matter (mean_sq ~ eps); candidate must still
    match the oracle — catches wrong eps placement that random-normal inputs hide."""
    case = next(c for c in make_cases() if c["name"] == case_name)
    inp = build_inputs(case)
    inp["q"] = inp["q"] * scale
    inp["k"] = inp["k"] * scale
    ref32 = torch_reference_fp32(inp)
    q_o, k_o = run_oracle(inp["q"].clone(), inp["k"].clone(), inp)
    try:
        q_c, k_c = run_candidate(inp["q"].clone(), inp["k"].clone(), inp)
    except NotImplementedError:
        pytest.skip("optimized_wrapper not implemented yet (stub)")
    assert _candidate_dispatch_path() == "cuda"
    _check_against_oracle(q_c, q_o, ref32[0], name=f"{case_name}/eps.q")
    _check_against_oracle(k_c, k_o, ref32[1], name=f"{case_name}/eps.k")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
