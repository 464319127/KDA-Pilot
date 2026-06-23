"""Valid speculative-tree input generation + an independent verification oracle.

Shared by ``adapter.py`` (benchmark/correctness case construction) and
``correctness.py`` (regression grid). The oracle is a pure-Python reimplementation
of the upstream ``VerifyTreeGreedy`` kernel (see ``../docs/baseline_source.md``);
it does not import, patch, or call SGLang.

Index conventions (matching the recovered kernel exactly):
  - ``retrive_index[b, col]`` is the GLOBAL / flattened node index. The canonical
    EAGLE layout (and what the upstream builder emits) is ``b * nd + col``.
  - ``retrive_next_token`` / ``retrive_next_sibling`` store ROW-LOCAL column slots
    (or -1), never global indices.
  - ``candidates`` is indexed row-locally (``b*nd + col``).
  - ``target_predict`` and ``predicts`` are indexed by the GLOBAL index.

All tree generators produce valid acyclic rooted trees: local slot 0 is the root,
every other slot has exactly one parent with a strictly smaller slot, children of a
parent are linked in increasing-slot order (first-linked sibling wins ties).
"""

from __future__ import annotations

import random
from typing import Any

import torch

VOCAB = 97  # small prime; keeps token ids int32-safe and reject offsets unambiguous

INT_INPUTS = (
    "candidates",
    "retrive_index",
    "retrive_next_token",
    "retrive_next_sibling",
    "target_predict",
)


def _links_from_parent(parent: list[int], nd: int) -> tuple[list[int], list[int]]:
    """Build row-local (next_token, next_sibling) from a parent array.

    parent[0] == -1 (root); parent[i] in [0, i-1] for i >= 1.
    """
    children: dict[int, list[int]] = {p: [] for p in range(nd)}
    for i in range(1, nd):
        children[parent[i]].append(i)
    next_token = [-1] * nd
    next_sibling = [-1] * nd
    for p in range(nd):
        ch = sorted(children[p])
        if ch:
            next_token[p] = ch[0]
            for k in range(len(ch) - 1):
                next_sibling[ch[k]] = ch[k + 1]
            next_sibling[ch[-1]] = -1
    return next_token, next_sibling


def _empty_row(nd: int) -> tuple[list[int], list[int]]:
    return [-1] * nd, [-1] * nd


def _fixed_upstream() -> dict[str, list[list[int]]]:
    """The exact fixture from sgl-kernel/tests/speculative/test_eagle_utils.py.

    Expected outputs (hand-verified, see docs/baseline_source.md):
      predicts          = [3,-1,-1,4,5,18, 11,-1,-1,-1,12,18]
      accept_index      = [[0,3,4,5],[6,10,11,-1]]
      accept_token_num  = [3,2]
    """
    return {
        "candidates": [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]],
        "retrive_index": [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
        "retrive_next_token": [[1, 2, -1, 4, 5, -1], [4, 2, 3, -1, 5, -1]],
        "retrive_next_sibling": [[-1, 3, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]],
        "target_predict": [[3, 18, 18, 4, 5, 18], [11, 18, 18, 18, 12, 18]],
    }


def build_inputs(
    mode: str,
    bs: int,
    nd: int,
    nss: int,
    *,
    seed: int,
    device: str | torch.device = "cpu",
    accept_prob: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return the five int64 input tensors for one workload case.

    ``mode`` is one of: random, fixed_upstream, all_accept, all_reject,
    sibling_tiebreak. ``retrive_index`` is always the canonical ``b*nd+col`` layout.
    """
    rng = random.Random(seed)

    if mode == "fixed_upstream":
        fx = _fixed_upstream()
        out = {k: torch.tensor(v, dtype=torch.int64, device=device) for k, v in fx.items()}
        return out

    retrive_index = [[b * nd + col for col in range(nd)] for b in range(bs)]
    candidates = [[0] * nd for _ in range(bs)]
    target = [[rng.randrange(VOCAB) for _ in range(nd)] for _ in range(bs)]
    next_token = [[-1] * nd for _ in range(bs)]
    next_sibling = [[-1] * nd for _ in range(bs)]

    def tgt(b: int, col: int) -> int:
        return target[b][col]

    for b in range(bs):
        if mode == "all_accept":
            # chain: 0 -> 1 -> 2 -> ... ; every edge matches its parent's target.
            parent = [-1] + list(range(0, nd - 1))
            nt, ns = _links_from_parent(parent, nd)
            next_token[b], next_sibling[b] = nt, ns
            candidates[b][0] = rng.randrange(VOCAB)
            for i in range(1, nd):
                candidates[b][i] = tgt(b, parent[i])
        elif mode == "all_reject":
            # star: every node is a child of the root; none matches target at root.
            parent = [-1] + [0] * (nd - 1)
            nt, ns = _links_from_parent(parent, nd)
            next_token[b], next_sibling[b] = nt, ns
            candidates[b][0] = rng.randrange(VOCAB)
            for i in range(1, nd):
                candidates[b][i] = (tgt(b, 0) + 1) % VOCAB
        elif mode == "sibling_tiebreak":
            # root has >=3 children; first two match -> first-linked must win.
            # slot1 also gets a matching child to exercise a 2-deep accept.
            assert nd >= 5, "sibling_tiebreak needs nd>=5"
            parent = [-1, 0, 0, 0, 1] + [1] * (nd - 5)
            nt, ns = _links_from_parent(parent, nd)
            next_token[b], next_sibling[b] = nt, ns
            candidates[b][0] = rng.randrange(VOCAB)
            candidates[b][1] = tgt(b, 0)            # match (should win)
            candidates[b][2] = tgt(b, 0)            # also match (must be skipped)
            candidates[b][3] = (tgt(b, 0) + 1) % VOCAB  # reject
            candidates[b][4] = tgt(b, 1)            # child of slot1: match -> 2-deep
            for i in range(5, nd):
                candidates[b][i] = (tgt(b, 1) + 1) % VOCAB
        else:  # "random"
            parent = [-1] + [rng.randint(0, i - 1) for i in range(1, nd)]
            nt, ns = _links_from_parent(parent, nd)
            next_token[b], next_sibling[b] = nt, ns
            candidates[b][0] = rng.randrange(VOCAB)
            for i in range(1, nd):
                p = parent[i]
                if rng.random() < accept_prob:
                    candidates[b][i] = tgt(b, p)
                else:
                    candidates[b][i] = (tgt(b, p) + 1) % VOCAB

    def t(x: list[list[int]]) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.int64, device=device)

    return {
        "candidates": t(candidates),
        "retrive_index": t(retrive_index),
        "retrive_next_token": t(next_token),
        "retrive_next_sibling": t(next_sibling),
        "target_predict": t(target),
    }


def make_outputs(
    bs: int,
    nd: int,
    nss: int,
    *,
    poison: int = -17,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Preallocate the three int32 output tensors, poison-filled."""
    return {
        "predicts": torch.full((bs * nd,), poison, dtype=torch.int32, device=device),
        "accept_index": torch.full((bs, nss), poison, dtype=torch.int32, device=device),
        "accept_token_num": torch.full((bs,), poison, dtype=torch.int32, device=device),
    }


@torch.no_grad()
def oracle_verify_tree_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    """Independent CPU oracle; mutates the three outputs in place.

    Faithful port of upstream ``VerifyTreeGreedy<int32_t, int64_t>``: only the
    accepted-path positions of ``predicts`` and ``accept_index`` are written, so
    untouched (poisoned) slots are preserved exactly as the kernel leaves them.
    """
    bs, nd = candidates.shape
    nss = accept_index.shape[1]

    cand = candidates.reshape(-1).tolist()
    ridx = retrive_index.reshape(-1).tolist()
    rnt = retrive_next_token.reshape(-1).tolist()
    rns = retrive_next_sibling.reshape(-1).tolist()
    tp = target_predict.reshape(-1).tolist()

    pred = predicts.reshape(-1).tolist()
    aidx = accept_index.reshape(-1).tolist()  # length bs*nss
    atn = accept_token_num.reshape(-1).tolist()

    for bx in range(bs):
        last_accepted = ridx[bx * nd + 0]
        aidx[bx * nss + 0] = last_accepted
        num_accepted = 0
        cur = 0
        for _j in range(1, nss):
            cur = rnt[bx * nd + cur]
            while cur != -1:
                draft_index = ridx[bx * nd + cur]
                draft_token = cand[bx * nd + cur]
                target_token = tp[last_accepted]
                if draft_token == target_token:
                    pred[last_accepted] = target_token
                    num_accepted += 1
                    aidx[bx * nss + num_accepted] = draft_index
                    last_accepted = draft_index
                    break
                cur = rns[bx * nd + cur]
            if cur == -1:
                break
        atn[bx] = num_accepted
        pred[last_accepted] = tp[last_accepted]

    predicts.copy_(torch.tensor(pred, dtype=predicts.dtype, device=predicts.device))
    accept_index.copy_(
        torch.tensor(aidx, dtype=accept_index.dtype, device=accept_index.device).reshape(bs, nss)
    )
    accept_token_num.copy_(
        torch.tensor(atn, dtype=accept_token_num.dtype, device=accept_token_num.device)
    )


def _selftest() -> None:
    # 1) Reproduce the upstream fixture exactly (poison with -1 like the upstream test).
    inp = build_inputs("fixed_upstream", 2, 6, 4, seed=0, device="cpu")
    out = make_outputs(2, 6, 4, poison=-1, device="cpu")
    oracle_verify_tree_greedy(
        out["predicts"], out["accept_index"], out["accept_token_num"], **inp
    )
    assert out["predicts"].tolist() == [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18], out["predicts"].tolist()
    assert out["accept_index"].tolist() == [[0, 3, 4, 5], [6, 10, 11, -1]], out["accept_index"].tolist()
    assert out["accept_token_num"].tolist() == [3, 2], out["accept_token_num"].tolist()
    print("OK upstream fixture: predicts/accept_index/accept_token_num match expected")

    # 2) Deterministic mode sanity.
    inp = build_inputs("all_reject", 4, 2, 2, seed=1)
    out = make_outputs(4, 2, 2, poison=-17)
    oracle_verify_tree_greedy(out["predicts"], out["accept_index"], out["accept_token_num"], **inp)
    assert out["accept_token_num"].tolist() == [0, 0, 0, 0], out["accept_token_num"].tolist()
    print("OK all_reject: accept_token_num all zero")

    inp = build_inputs("all_accept", 2, 4, 4, seed=2)
    out = make_outputs(2, 4, 4, poison=-17)
    oracle_verify_tree_greedy(out["predicts"], out["accept_index"], out["accept_token_num"], **inp)
    # chain of nd=4 -> at most nss-1=3 accepted; depth allows 3.
    assert out["accept_token_num"].tolist() == [3, 3], out["accept_token_num"].tolist()
    print("OK all_accept (nd=4,nss=4): 3 accepted each")

    inp = build_inputs("sibling_tiebreak", 1, 5, 3, seed=3)
    out = make_outputs(1, 5, 3, poison=-17)
    oracle_verify_tree_greedy(out["predicts"], out["accept_index"], out["accept_token_num"], **inp)
    # first matching sibling (slot1, global 1) then its matching child (slot4, global 4).
    assert out["accept_token_num"].tolist() == [2], out["accept_token_num"].tolist()
    assert out["accept_index"].tolist() == [[0, 1, 4]], out["accept_index"].tolist()
    print("OK sibling_tiebreak: first-linked sibling wins, 2-deep accept [0,1,4]")

    print("ALL ORACLE SELF-TESTS PASSED")


if __name__ == "__main__":
    _selftest()
