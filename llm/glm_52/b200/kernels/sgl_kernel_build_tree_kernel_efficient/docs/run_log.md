# Run Log

## Environment
- Remote host: `ion-b200` (`innomatrix-us-adc-smb200-0003`)
- Docker container: `sglang_bbuf`
- Target GPU: NVIDIA B200, **GPU id 6** (`REMOTE_GPU_ID=6`, pinned via `CUDA_VISIBLE_DEVICES=6` for every build / correctness / benchmark / probe command)
- Task source commit (this worktree base): `47114cdca5ee630ce599c91abce17b8fb95c3a4d`
- Upstream baseline commit: `7e6587c94a1d0305815a14067c5d3cc02a9b0f36`
- Candidate sha256: `e6668cc557c93ec43351e9a49266a00b8759550f4aa30891ddfa63431f5d5a0a` (Round 4; was 70e6a388 in Round 2 — `candidate.cu` predicate now enforces the captured 2-D retrieve shape)

## Round 4 (captured 2-D retrieve shape — functional verification)
GPU 6 was **occupied by another user's job** this round (foreign pid holding ~157.7 GB,
util 5–72%): BEFORE `6, NVIDIA B200, 77 %, 157680 MiB / 183359 MiB`; AFTER
`6, NVIDIA B200, 69 %, 157684 MiB / 183359 MiB`. Per the GPU-idle policy, **no timing
was measured on the busy GPU and no GPU switch was made**. The non-timed, low-memory
functional checks (build, correctness, route) ran while sharing:
- Rebuild OK (build cache cleared); `python bench/correctness.py` → **691 cases, 0
  failures**, route==1 for all bs 1..10 on the 2-D `[bs,NV]` retrieve shape (incl.
  bs=10 `[10,2]`), route==0 for the 5 off-domain cases.
- Benchmark + floor_probe **deferred** (busy GPU). Timing is rank-invariant (see
  `docs/results.md`), so the Round-2 numbers remain valid for the 2-D shape; the
  rebuilt 2-D module is staged on the remote to re-time once GPU 6 is idle.

## GPU idleness (performance data valid only when idle)
### Before
```
6, NVIDIA B200, 0 %, 4 MiB, 183359 MiB
```
### After
```
6, NVIDIA B200, 0 %, 4 MiB, 183359 MiB   (idle; 4 MiB residual)
```

## Software versions
- PyTorch + CUDA 13 (cu13); tvm-ffi build; GPU arch `sm_100` (Blackwell B200).
- ABI/build: `tvm_ffi.cpp.load` — baseline + candidate compiled together into one
  TVM-FFI module (now also exporting `build_tree_candidate_route`), symmetric
  `-std=c++17 -O3 -gencode …sm_100`, torch ldlibs, no fast-math.

## Commands run (exact, all pinned to GPU 6)
```
# build (JIT via tvm_ffi.cpp.load, untimed)
CUDA_VISIBLE_DEVICES=6 python -c "import sys; sys.path.insert(0,'bench'); import build_ext; build_ext.get_ext()"
# correctness -> 691 cases, 0 failures; route assertions PASS (fast-path all bs 1..10, fallback all 5)
CUDA_VISIBLE_DEVICES=6 python bench/correctness.py
# benchmark (all 183 production rows; candidate genuinely runs for bs>1) -> production geomean 0.9839
CUDA_VISIBLE_DEVICES=6 python bench/benchmark.py --device cuda:0 --warmup-runs 10 --num-trials 7 \
  --inner-iterations-min 1 --inner-iterations-max 4096 --target-sample-us 1000 --out bench/results.jsonl
# controlled probe (all bs 1..10) + wrapper-inclusive diagnostic
CUDA_VISIBLE_DEVICES=6 python bench/floor_probe.py | tee bench/floor_probe_out.txt
```

## Notes
- ROUND 2 fixed the Round-1 dispatch bug: `bte::is_contiguous` treated the captured
  `parent_list [bs,0]` as non-contiguous for bs>1, so the candidate silently fell
  back to baseline for 158/183 rows. Fixed (zero-element tensor → contiguous) and
  PROVEN via the `build_tree_candidate_route` diagnostic + correctness route
  assertions (route==1 for all bs 1..10 incl. bs=10 `[10,0]`; route==0 for the 5
  off-domain cases).
- With the candidate genuinely exercised, the verdict is a definitive NO-GO:
  correctness 691/0; official production geomean 0.9839 (0 clean wins, 22 real
  regressions, 161 ties); controlled probe every bs a tie (geomean 0.9877); empty
  floor ~4.4–6.7µs vs op ~10–12µs (host/launch-bound). See `docs/results.md`.
- Raw `bench/results.jsonl` + remote build log kept local/unstaged; controlled-probe
  output saved to `bench/floor_probe_out.txt`.
