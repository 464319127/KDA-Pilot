"""Adversarial randomized correctness stress for the candidate vs the recovered baseline.

The candidate replicates upstream moeTopK's `-1.f`/index-0 argmax sentinel, so it must match the
recovered baseline EXACTLY on every row here — including the degenerate all-scores-<-1 corner where
the sentinel wins (this input is inside the host-only route predicate, which cannot inspect bias
values, so bit-faithfulness is mandatory, not range-restricted). Sweeps bias scales, extreme logits,
and the sentinel corner; all rows are required passes.

Run on the GPU: CUDA_VISIBLE_DEVICES=<id> PYTHONPATH=. python stress_adversarial.py
"""

import torch

import build_ext

DEVICE = torch.device("cuda")
N, E, K = 64, 288, 8


def run(g, b):
    wb = torch.empty((N, K), dtype=torch.float32, device=DEVICE)
    ib = torch.empty((N, K), dtype=torch.int32, device=DEVICE)
    wc = torch.empty_like(wb)
    ic = torch.empty_like(ib)
    for t in (wb, wc):
        t.fill_(float("nan"))
    for t in (ib, ic):
        t.fill_(-17)
    build_ext.baseline(wb, ib, g, 1, b)
    build_ext.candidate(wc, ic, g, 1, b)
    torch.cuda.synchronize()
    ids_eq = torch.equal(ib.to(torch.int64), ic.to(torch.int64))
    w_ok = torch.allclose(wc.float(), wb.float(), atol=1e-5, rtol=1e-5)
    return ids_eq, w_ok


def main() -> int:
    configs = []
    for scale in (0.1, 0.5, 1.0, 2.0, 3.0):
        for seed in range(5):
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            g = torch.randn((N, E), dtype=torch.float32, device=DEVICE, generator=gen)
            b = torch.randn((E,), dtype=torch.float32, device=DEVICE, generator=gen) * scale
            configs.append((f"bias_scale={scale} seed={seed}", g, b, True))
    gen = torch.Generator(device=DEVICE).manual_seed(99)
    configs.append(("extreme_logits_x30", torch.randn((N, E), dtype=torch.float32, device=DEVICE, generator=gen) * 30.0,
                    torch.randn((E,), dtype=torch.float32, device=DEVICE, generator=gen), True))
    # Sentinel corner: all scores < -1, so upstream moeTopK's -1.f/index-0 sentinel wins every
    # round. The candidate replicates that sentinel exactly, so it MUST match the baseline here too
    # (required pass) — not a documented divergence. This input is inside the host-only route
    # predicate (which cannot inspect bias values), so bit-faithfulness here is mandatory.
    configs.append(("all_neg_bias_-5 (sentinel corner, required)",
                    torch.randn((N, E), dtype=torch.float32, device=DEVICE, generator=gen),
                    torch.full((E,), -5.0, dtype=torch.float32, device=DEVICE), True))

    realistic_pass = realistic_total = 0
    for name, g, b, realistic in configs:
        ie, wok = run(g, b)
        ok = ie and wok
        if realistic:
            realistic_total += 1
            realistic_pass += int(ok)
        tag = "OK  " if ok else "DIFF"
        print(f"{tag} {name}: ids_eq={ie} weights_ok={wok}")
    print(f"\nrealistic configs: {realistic_pass}/{realistic_total} candidate==baseline (exact ids + fp32 weights)")
    return 0 if realistic_pass == realistic_total else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
