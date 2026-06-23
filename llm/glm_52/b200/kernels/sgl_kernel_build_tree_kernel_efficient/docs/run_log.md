# Run Log

## Environment
- Remote host: `ion-b200` (`innomatrix-us-adc-smb200-0003`)
- Docker container: `sglang_bbuf`
- Target GPU: NVIDIA B200, **GPU id 6** (`REMOTE_GPU_ID=6`, pinned via `CUDA_VISIBLE_DEVICES=6` for every build / correctness / benchmark / probe command)
- Task source commit (this worktree base): `47114cdca5ee630ce599c91abce17b8fb95c3a4d`
- Upstream baseline commit: `7e6587c94a1d0305815a14067c5d3cc02a9b0f36`

## GPU idleness (performance data valid only when idle)
### Before
```
GPU 6, NVIDIA B200, util 0 %, mem 0 MiB / 183359 MiB
```
### After
```
GPU 6, NVIDIA B200, util 0 %, mem 4 MiB / 183359 MiB   (idle)
```

## Software versions
- PyTorch: torch + CUDA 13 (cu13); tvm-ffi: 0.1.9
- CUDA / nvcc: 13; GPU architecture `sm_100` (Blackwell B200)
- ABI/build: `tvm_ffi.cpp.load` — baseline + candidate compiled together into one
  TVM-FFI module, symmetric `-std=c++17 -O3 -gencode …sm_100`, torch ldlibs, no
  fast-math. Candidate sha256 `32abc09611f5a475370f07dac71b0e3b84468f8c13e60382c08db0afec3b647a`.

## Commands run (exact, all pinned to GPU 6)
```
# build (JIT via tvm_ffi.cpp.load, untimed)
CUDA_VISIBLE_DEVICES=6 python -c "import sys; sys.path.insert(0,'bench'); import build_ext; build_ext.get_ext()"
# correctness  -> ran 691 cases; 0 failures; CORRECTNESS: PASS
CUDA_VISIBLE_DEVICES=6 python bench/correctness.py
# benchmark (all 187 workloads / 183 production) -> production geomean 0.9932
CUDA_VISIBLE_DEVICES=6 python bench/benchmark.py --device cuda:0 --warmup-runs 10 --num-trials 7 \
  --inner-iterations-min 1 --inner-iterations-max 4096 --target-sample-us 1000 --out bench/results.jsonl
# controlled probe (all bs 1..10) + wrapper-inclusive diagnostic
CUDA_VISIBLE_DEVICES=6 python bench/floor_probe.py | tee bench/floor_probe_out.txt
```

## Notes
- ROUND 1 reran everything after Codex review: TVM-FFI ABI (was pybind), full 183
  captured production workloads (was 10), no-wrap output-buffer ring (was a ring
  that could wrap), controlled probe extended to ALL bs incl. 7 and 9, and a
  measured (not inferred) wrapper-inclusive diagnostic.
- Verdict: evidence-backed NO-GO — candidate is correct (691/0) but a statistical
  tie (controlled geomean 0.9854, official 0.9932; every bs a tie). Named bound:
  ~3.35µs launch floor + ~6µs 8-tensor TensorView marshalling dominate a <0.1µs
  kernel body. See `docs/results.md`.
- Raw `bench/results.jsonl` + remote build log kept local/unstaged; controlled-probe
  output saved to `bench/floor_probe_out.txt`.
