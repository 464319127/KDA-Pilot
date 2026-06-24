"""Runtime probe of the recovered baseline ABI: record the actual return value, the
in-place mutation set, output shape/dtype/stride/device/contiguity, and current-stream behavior —
rather than inferring them from the op schema alone. Paste the transcript into docs/baseline_source.md.

Run on the GPU: CUDA_VISIBLE_DEVICES=<id> PYTHONPATH=. python probe_baseline.py
"""

import torch

import build_ext

DEVICE = torch.device("cuda")
N, E, K = 4, 288, 8


def fingerprint(t):
    return (t.data_ptr(), tuple(t.shape), str(t.dtype), tuple(t.stride()),
            str(t.device), t.is_contiguous())


def main() -> int:
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    gating = torch.randn((N, E), dtype=torch.float32, device=DEVICE, generator=gen)
    bias = torch.randn((E,), dtype=torch.float32, device=DEVICE, generator=gen)
    w = torch.empty((N, K), dtype=torch.float32, device=DEVICE)
    idx = torch.empty((N, K), dtype=torch.int32, device=DEVICE)
    w.fill_(float("nan"))
    idx.fill_(-17)

    gating_before = gating.clone()
    w_ptr0, idx_ptr0, g_ptr0 = w.data_ptr(), idx.data_ptr(), gating.data_ptr()

    ret = build_ext.baseline(w, idx, gating, 1, bias)
    torch.cuda.synchronize()

    print("=== topk_sigmoid baseline runtime probe (N=4, E=288, K=8, fp32, renormalize=True) ===")
    print(f"python_return: {ret!r}  (is None: {ret is None})")
    print(f"arg0 topk_weights data_ptr unchanged: {w.data_ptr() == w_ptr0}; "
          f"contents changed vs poison: {bool(torch.isfinite(w).all().item())}")
    print(f"arg1 topk_indices data_ptr unchanged: {idx.data_ptr() == idx_ptr0}; "
          f"contents changed vs poison: {bool((idx != -17).any().item())}; "
          f"all valid experts [0,{E}): {bool(((idx >= 0) & (idx < E)).all().item())}")
    print(f"arg2 gating_output data_ptr unchanged: {gating.data_ptr() == g_ptr0}; "
          f"contents MUTATED: {not torch.equal(gating, gating_before)}  (expect False = read-only)")
    print(f"out topk_weights: {fingerprint(w)}")
    print(f"out topk_indices: {fingerprint(idx)}")

    # Current-stream behavior: enqueue on a NON-default stream; an event recorded on that stream
    # after the call must complete once only that stream is synchronized -> the launch used the
    # current (non-default) stream via at::cuda::getCurrentCUDAStream().
    s = torch.cuda.Stream()
    w2 = torch.empty((N, K), dtype=torch.float32, device=DEVICE)
    idx2 = torch.empty((N, K), dtype=torch.int32, device=DEVICE)
    w2.fill_(float("nan"))
    idx2.fill_(-17)
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        build_ext.baseline(w2, idx2, gating, 1, bias)
        ev = torch.cuda.Event()
        ev.record(s)
    s.synchronize()
    print(f"non-default-stream event complete after only that stream's sync: {ev.query()}  "
          f"(True -> launch used the current CUDA stream)")
    print(f"outputs finite after non-default-stream sync: {bool(torch.isfinite(w2).all().item())}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
