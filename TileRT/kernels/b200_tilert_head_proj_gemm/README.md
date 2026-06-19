# b200_tilert_head_proj_gemm

TileRT `HeadProjExecutorImpl` — the DeepSeek-V3.2 LM-head GEMM (decode).
Optimize a B200 CUDA kernel to match TileRT's measured 39.2µs @ 78.4% HBM.
