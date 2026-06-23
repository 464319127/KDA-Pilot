"""Exact-match correctness gate for `verify_tree_greedy`.

For every frozen workload (and several seeds each) this checks, with EXACT
integer/structural equality on all three outputs (`predicts`, `accept_index`,
`accept_token_num`) including untouched/poisoned slots:

    candidate (CUDA)  ==  baseline (CUDA)  ==  independent CPU oracle

Poisoned output buffers are filled before every run so stale/partial-write and
skipped-kernel bugs are visible. The grid covers the captured regime plus synthetic
full-accept / partial-accept / full-reject / sibling-tie-break trees and a
`bs>CAP, nd>2` row that must route through the candidate's baseline fallback.

Run on the target GPU:  python bench/correctness.py
(CPU-only: runs the independent-oracle self-tests; the CUDA exact-match needs a GPU.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import adapter  # noqa: E402
import tree_inputs  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)


def _cmp(name: str, got: torch.Tensor, ref: torch.Tensor) -> str | None:
    got = got.cpu()
    if got.shape != ref.shape:
        return f"{name}: shape {tuple(got.shape)} vs {tuple(ref.shape)}"
    if got.dtype != ref.dtype:
        return f"{name}: dtype {got.dtype} vs {ref.dtype}"
    if not torch.equal(got, ref):
        nmm = int((got != ref).sum().item())
        return (f"{name}: {nmm} mismatch(es); got={got.flatten().tolist()[:16]} "
                f"ref={ref.flatten().tolist()[:16]}")
    return None


def _run_case(workload: dict, seed: int, device: torch.device) -> list[str]:
    bs, nd, nss = adapter._shapes(workload)
    tree = workload.get("tree", {"mode": "random"})
    mode = tree.get("mode", "random")
    ap = float(tree.get("accept_prob", 0.5))

    inp = tree_inputs.build_inputs(mode, bs, nd, nss, seed=seed, device=device, accept_prob=ap)
    inputs = tuple(inp[k] for k in adapter._INPUT_ORDER)

    base = tree_inputs.make_outputs(bs, nd, nss, poison=adapter._POISON, device=device)
    cand = tree_inputs.make_outputs(bs, nd, nss, poison=adapter._POISON, device=device)
    base_t = tuple(base[k] for k in adapter._OUTPUT_ORDER)
    cand_t = tuple(cand[k] for k in adapter._OUTPUT_ORDER)

    adapter.call_baseline(workload, inputs, base_t)
    adapter.call_candidate(workload, inputs, cand_t)
    torch.cuda.synchronize()

    # Independent oracle on CPU copies of the same inputs and the same poison.
    inp_cpu = {k: v.cpu() for k, v in inp.items()}
    orc = tree_inputs.make_outputs(bs, nd, nss, poison=adapter._POISON, device="cpu")
    tree_inputs.oracle_verify_tree_greedy(
        orc["predicts"], orc["accept_index"], orc["accept_token_num"], **inp_cpu
    )

    fails = []
    for name, bt in zip(adapter._OUTPUT_ORDER, base_t):
        m = _cmp(f"baseline-vs-oracle {name}", bt, orc[name])
        if m:
            fails.append(m)
    for name, ct in zip(adapter._OUTPUT_ORDER, cand_t):
        m = _cmp(f"candidate-vs-oracle {name}", ct, orc[name])
        if m:
            fails.append(m)
    return fails


def _upstream_fixture_gpu(device: torch.device) -> list[str]:
    """ext baseline/candidate on the upstream fixture must match the known outputs."""
    inp = tree_inputs.build_inputs("fixed_upstream", 2, 6, 4, seed=0, device=device)
    inputs = tuple(inp[k] for k in adapter._INPUT_ORDER)
    exp_pred = [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18]
    exp_aidx = [[0, 3, 4, 5], [6, 10, 11, -1]]
    exp_atn = [3, 2]
    fails = []
    for side, call in (("baseline", adapter.call_baseline), ("candidate", adapter.call_candidate)):
        out = tree_inputs.make_outputs(2, 6, 4, poison=-1, device=device)  # -1 like the upstream test
        ot = tuple(out[k] for k in adapter._OUTPUT_ORDER)
        call({"shapes": {"bs": 2, "num_draft_tokens": 6, "num_spec_step": 4}}, inputs, ot)
        torch.cuda.synchronize()
        if out["predicts"].cpu().tolist() != exp_pred:
            fails.append(f"{side} upstream predicts {out['predicts'].cpu().tolist()}")
        if out["accept_index"].cpu().tolist() != exp_aidx:
            fails.append(f"{side} upstream accept_index {out['accept_index'].cpu().tolist()}")
        if out["accept_token_num"].cpu().tolist() != exp_atn:
            fails.append(f"{side} upstream accept_token_num {out['accept_token_num'].cpu().tolist()}")
    return fails


def main() -> int:
    workloads = json.loads((_HERE / "workloads.json").read_text())

    if not torch.cuda.is_available():
        print("[correctness] CUDA unavailable — running independent-oracle self-tests only.")
        tree_inputs._selftest()
        print("[correctness] Oracle self-tests passed. Full exact-match grid requires a GPU.")
        return 0

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"[correctness] device={torch.cuda.get_device_name(device)}")

    total = 0
    failed = 0

    fx = _upstream_fixture_gpu(device)
    total += 1
    if fx:
        failed += 1
        print("FAIL  upstream_fixture")
        for m in fx:
            print("   -", m)
    else:
        print("PASS  upstream_fixture (baseline & candidate match known outputs)")

    for w in workloads:
        wid = w["id"]
        case_fails = []
        for seed in SEEDS:
            case_fails += [f"seed={seed}: {m}" for m in _run_case(w, seed, device)]
        total += 1
        if case_fails:
            failed += 1
            print(f"FAIL  {wid}")
            for m in case_fails[:8]:
                print("   -", m)
        else:
            print(f"PASS  {wid}  (baseline==candidate==oracle over {len(SEEDS)} seeds)")

    print(f"\n[correctness] {total - failed}/{total} checks passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
