# TileRT reference measurement — head_proj (LM-head GEMM)

Measured on NVIDIA B200 from the shipped `libtilert_dsv32.so` (tilert 0.1.4).

## How
```python
import tilert, torch
tilert.load_backend("deepseek_v3_2"); torch.ops.tilert.tilert_init_op()
V,K=16160,7168
h=torch.randn(1,1,K,device="cuda",dtype=torch.bfloat16)/K**0.5
W=torch.randn(V,K,device="cuda",dtype=torch.bfloat16)/K**0.5
# TileRT wants the "native bf16 warp gemv" swizzle:
Wc=W.reshape(V//16,16,K//1024,1024).transpose(1,2).reshape(V//16*K//1024,16,1024).contiguous()
out=torch.zeros(1,1,V,device="cuda",dtype=torch.float32)
pl=torch.zeros(66,148,16,dtype=torch.uint64,device="cuda")
torch.ops.tilert.head_proj_op(h,Wc,out,"deepseek_v3_2","general",pl)   # matches h@W.T, rel ~5e-3
```
ncu:
```
ncu --clock-control none --kernel-name regex:HeadProj --launch-count 1 \
    --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum \
    python run_once.py
```

## Result
```
HeadProjExecutorImpl<DefaultSchedule, 4, 40960, 1, 1, 3>(GlbArgs<8,2>)
  grid (148,1,1) x block (384,1,1)
  dram__bytes.sum            235.23 MB
  gpu__time_duration.avg     39.17 us
  dram__throughput           78.36 % of peak  (= 6.0 TB/s on B200)
```
Target for the CUDA candidate: ≤ ~39 µs / ≥ ~78% HBM on seq ∈ {1,2,4}.
