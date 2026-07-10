"""Bring-up smoke test: single-process NVLS workspace + flashinfer ORIGINAL.

Validates, before any benchmarking:
  1. the workspace assembly (multicast create/bind/map) succeeds on this box,
  2. the flashinfer oneshot fused kernel runs eagerly on all 8 ranks against
     our workspace and passes the composed fp32 oracle at the contract bf16
     tolerance (atol 7e-2 / rtol 2e-2; max-rel is reported as a diagnostic —
     its floor is one bf16 ulp ~3.9e-3),
  3. the Lamport flag rotation advances across consecutive calls,
  4. a second round on rotated buffers stays correct.

Run under a hard timeout (a hang here means the multicast fan-out assumption
failed and the Lamport spin never completes).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_ROOT)

from bench.correctness import contract_allclose, fp32_oracle, poison_, rel_err  # noqa: E402
from bench.sp_nvls_workspace import SpNvlsWorkspace  # noqa: E402
from baseline import fi_original  # noqa: E402


def make_io(world: int, T: int, H: int, seed: int):
    torch.manual_seed(seed)
    gamma_cpu = torch.randn(H, dtype=torch.bfloat16)
    residual_cpu = torch.randn(T, H, dtype=torch.bfloat16)
    xs, residuals, gammas, outs, resouts = [], [], [], [], []
    for i in range(world):
        dev = f"cuda:{i}"
        torch.cuda.set_device(i)
        xs.append(torch.randn(T, H, dtype=torch.bfloat16, device=dev))
        residuals.append(residual_cpu.to(dev))  # replicated, byte-identical
        gammas.append(gamma_cpu.to(dev))
        outs.append(poison_(torch.empty(T, H, dtype=torch.bfloat16, device=dev)))
        resouts.append(poison_(torch.empty(T, H, dtype=torch.bfloat16, device=dev)))
    return xs, residuals, gammas, outs, resouts


def run_round(ws, xs, residuals, gammas, outs, resouts, eps, pdl=False):
    world = ws.world_size
    for i in range(world):
        torch.cuda.set_device(i)
        fi_original.launch(
            xs[i], outs[i], residuals[i], resouts[i], gammas[i], ws.ranks[i], eps, pdl
        )
    for i in range(world):
        torch.cuda.synchronize(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    T, H, world, eps = args.tokens, args.hidden, args.world, args.eps

    print(f"[smoke] building SpNvlsWorkspace world={world} ...", flush=True)
    t0 = time.time()
    ws = SpNvlsWorkspace(world_size=world)
    print(f"[smoke] workspace ok in {time.time() - t0:.1f}s "
          f"(total {ws.total_bytes} B, per-buffer {ws.buffer_size_bytes} B)", flush=True)

    xs, residuals, gammas, outs, resouts = make_io(world, T, H, args.seed)

    print("[smoke] round 1 (flags should rotate 0 -> 1) ...", flush=True)
    run_round(ws, xs, residuals, gammas, outs, resouts, eps)
    oracle = fp32_oracle([x.to("cuda:0") for x in xs], residuals[0].to("cuda:0"),
                         gammas[0].to("cuda:0"), eps)
    worst_out = max(rel_err(outs[i], oracle.out.to(outs[i].device)) for i in range(world))
    worst_res = max(
        rel_err(resouts[i], oracle.residual_out.to(resouts[i].device)) for i in range(world)
    )
    flags0 = ws.ranks[0].buffer_flags.cpu().tolist()
    print(f"[smoke] round1 max-rel(out)={worst_out:.3e} max-rel(res)={worst_res:.3e} "
          f"(bf16 ulp floor ~3.9e-3) flags(rank0)={flags0}", flush=True)
    ok1 = all(
        contract_allclose(outs[i], oracle.out.to(outs[i].device))[0]
        and contract_allclose(resouts[i], oracle.residual_out.to(resouts[i].device))[0]
        for i in range(world)
    )

    # cross-rank consistency: all ranks must agree bitwise with rank 0
    xrank_ok = all(
        torch.equal(outs[i].cpu().view(torch.int16), outs[0].cpu().view(torch.int16))
        for i in range(world)
    )

    print("[smoke] round 2 on rotated buffers ...", flush=True)
    for i in range(world):
        poison_(outs[i]); poison_(resouts[i])
    run_round(ws, xs, residuals, gammas, outs, resouts, eps)
    worst_out2 = max(rel_err(outs[i], oracle.out.to(outs[i].device)) for i in range(world))
    flags1 = ws.ranks[0].buffer_flags.cpu().tolist()
    print(f"[smoke] round2 max-rel(out)={worst_out2:.3e} flags(rank0)={flags1}", flush=True)
    ok2 = all(
        contract_allclose(outs[i], oracle.out.to(outs[i].device))[0] for i in range(world)
    )

    rotated = flags0[0] != flags1[0]
    verdict = ok1 and ok2 and xrank_ok and rotated
    print(f"[smoke] cross-rank bitwise agree: {xrank_ok}; flag rotation: {rotated}")
    print(f"[smoke] {'PASS' if verdict else 'FAIL'}")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
