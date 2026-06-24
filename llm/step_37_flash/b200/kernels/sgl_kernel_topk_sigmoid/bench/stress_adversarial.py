"""Adversarial randomized correctness stress for the candidate vs the recovered baseline.

Closes the gap flagged by the independent Codex cross-check: the baseline moeTopK uses a `-1.f`
sentinel as the initial argmax value, so if a token's `sigmoid(logit)+bias` scores fall below -1
for nearly all experts, the baseline degenerates (the sentinel wins), while the candidate (genuine
top-k) does not. This sweeps bias scales and extreme logits to show candidate==baseline across the
realistic range, and probes the pathological all-scores-<-1 corner to document the exact boundary.

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
    # Pathological corner: all scores < -1 -> baseline's -1.f sentinel degenerates. Expected to
    # diverge; this row documents the boundary and is NOT in the realistic Step-3.7 bias range.
    configs.append(("all_neg_bias_-5 (sentinel corner, expected DIFF)",
                    torch.randn((N, E), dtype=torch.float32, device=DEVICE, generator=gen),
                    torch.full((E,), -5.0, dtype=torch.float32, device=DEVICE), False))

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
