# TileRT kernel benchmark method
Target = the measured TileRT op latency (and HBM %) in each task's
`config.toml [reference]` / `docs/tilert_reference.md`, obtained by calling the
real `libtilert_dsv32.so` op and profiling with ncu (gpu__time_duration.avg,
dram__throughput.avg.pct_of_peak_sustained_elapsed) on B200. A candidate "matches
TileRT" when its median latency over `num_trials` is within noise of the TileRT
number on the same shape, at matching correctness. seq ∈ {1,2,4} (decode/MTP).
Note: some kernels currently list a profiler per-call average (marked); isolated
ncu numbers will replace them once GPU access is restored.
