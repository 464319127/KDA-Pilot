"""A/B bit-compare fi vs jit on adversarial bf16 value classes.

randn-based inputs never produce subnormals, signed zeros, or extreme
magnitudes — exactly the classes where build-flag asymmetries (FTZ /
fast-math) between two compilations of the SAME kernel source diverge.
Serving activations DO contain them. This script laces real-shaped inputs
with the full bf16 value zoo and bit-compares the two implementations.
"""

from __future__ import annotations

import os
import sys

import torch

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_ROOT)

from bench.correctness import bitwise_equal, poison_  # noqa: E402
from bench.sp_nvls_workspace import SpNvlsWorkspace  # noqa: E402

WORLD, H, EPS = 8, 6144, 1e-5


def _load_impl(name: str):
    if name == "fi":
        from baseline import fi_original

        return fi_original
    if name == "jit":
        from solution import jit_port

        return jit_port
    if name == "opt":
        from solution import jit_port_opt

        return jit_port_opt
    raise ValueError(name)

ZOO_U16 = [
    0x8000,  # -0.0
    0x0000,  # +0.0
    0x0001,  # +min subnormal
    0x8001,  # -min subnormal
    0x007F,  # +max subnormal
    0x807F,  # -max subnormal
    0x0080,  # +min normal
    0x8080,  # -min normal
    0x7F00,  # large +
    0xFF00,  # large -
    0x3F80,  # 1.0
    0xBF80,  # -1.0
    0x0040,  # mid subnormal
    0x8740,  # small negative normal
]


def lace(t: torch.Tensor, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    flat = t.view(torch.int16).flatten()
    n = flat.numel()
    # scatter each zoo value into ~1% of positions
    for k, v in enumerate(ZOO_U16):
        idx = torch.randperm(n, generator=g)[: max(1, n // 100)]
        sv = torch.tensor(v, dtype=torch.uint16).view(torch.int16)
        flat[idx.to(flat.device)] = sv.to(flat.device)


def run_case(ws, impls, T: int, seed: int, pdl: bool) -> int:
    torch.manual_seed(seed)
    gamma_cpu = torch.randn(H, dtype=torch.bfloat16)
    lace(gamma_cpu, seed + 99)
    residual_cpu = torch.randn(T, H, dtype=torch.bfloat16)
    lace(residual_cpu, seed + 7)
    xs, res, gam = [], [], []
    for i in range(WORLD):
        torch.cuda.set_device(i)
        x = torch.randn(T, H, dtype=torch.bfloat16, device=f"cuda:{i}")
        lace(x, seed + i)
        xs.append(x)
        res.append(residual_cpu.to(f"cuda:{i}"))
        gam.append(gamma_cpu.to(f"cuda:{i}"))
    outs = {}
    for name, impl in impls:
        ws.reset()
        o = [poison_(torch.empty(T, H, dtype=torch.bfloat16, device=f"cuda:{i}"))
             for i in range(WORLD)]
        r = [poison_(torch.empty(T, H, dtype=torch.bfloat16, device=f"cuda:{i}"))
             for i in range(WORLD)]
        for i in range(WORLD):
            torch.cuda.set_device(i)
            impl.launch(xs[i], o[i], res[i], r[i], gam[i], ws.ranks[i], EPS, pdl)
        for i in range(WORLD):
            torch.cuda.synchronize(i)
        outs[name] = (o, r)
    a, b = impls[0][0], impls[1][0]
    bad = 0
    for i in range(WORLD):
        for kind, ta, tb in (("out", outs[a][0][i], outs[b][0][i]),
                             ("res", outs[a][1][i], outs[b][1][i])):
            eq, count, first = bitwise_equal(ta, tb)
            if not eq:
                bad += count
                if count:
                    fa = ta.flatten().view(torch.int16)[first].item() & 0xFFFF
                    fb = tb.flatten().view(torch.int16)[first].item() & 0xFFFF
                    print(f"  T={T} rank{i} {kind}: {count} mismatches, "
                          f"first idx {first}: fi=0x{fa:04X} jit=0x{fb:04X}")
    return bad


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", default="fi,jit", help="comma pair, e.g. fi,opt")
    args = ap.parse_args()
    names = args.impls.split(",")
    assert len(names) == 2
    impls = [(n, _load_impl(n)) for n in names]

    ws = SpNvlsWorkspace(world_size=WORLD)
    total = 0
    try:
        for T in (6, 1):
            for pdl in (False, True):
                for seed in (11, 22, 33):
                    bad = run_case(ws, impls, T, seed, pdl)
                    tag = "MATCH" if bad == 0 else f"DIVERGE({bad})"
                    print(f"[zoo] {names[0]} vs {names[1]} T={T} pdl={int(pdl)} "
                          f"seed={seed}: {tag}")
                    total += bad
    finally:
        ws.destroy()
    print(f"[zoo] {names[0]} vs {names[1]} TOTAL mismatched elements: {total} -> "
          f"{'BIT-EXACT on value zoo' if total == 0 else 'VALUE-CLASS DIVERGENCE CONFIRMED'}")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
