"""Minimal NCU profiling target: run the candidate fused kernel and the baseline's two
kernels (moeSigmoid + moeTopK) once each on a large-N row, after warmup, so Nsight Compute
can attribute achieved DRAM/compute throughput + stalls. Raw .ncu-rep stays local (excluded
from the PR); only the summarized metrics go into docs/results.md.

  CUDA_VISIBLE_DEVICES=<idle> ncu --set basic --launch-skip 6 --launch-count 3 \
    -o /tmp/topk_ncu --force-overwrite python profile_ncu.py
(2 warmup iters = 6 launches [candidate=1, baseline=2 each] -> skip 6, profile the next 3.)
"""

import torch

import build_ext

N, E, K = 16474, 288, 8
device = torch.device("cuda")
g = torch.randn((N, E), dtype=torch.float32, device=device)
b = torch.randn((E,), dtype=torch.float32, device=device)
wc = torch.empty((N, K), dtype=torch.float32, device=device)
ic = torch.empty((N, K), dtype=torch.int32, device=device)
wb = torch.empty((N, K), dtype=torch.float32, device=device)
ib = torch.empty((N, K), dtype=torch.int32, device=device)

# warmup (2 iters): candidate (1 kernel) + baseline (moeSigmoid + moeTopK = 2 kernels)
for _ in range(2):
    build_ext.candidate(wc, ic, g, 1, b)
    build_ext.baseline(wb, ib, g, 1, b)
torch.cuda.synchronize()

# profiled region (3 launches): candidate fused, then baseline's two kernels
build_ext.candidate(wc, ic, g, 1, b)
build_ext.baseline(wb, ib, g, 1, b)
torch.cuda.synchronize()
print("profile target done")
