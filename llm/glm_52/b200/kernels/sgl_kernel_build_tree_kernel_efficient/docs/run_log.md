# Run Log

## Environment
- Remote host: `ion-b200` (`innomatrix-us-adc-smb200-0003`)
- Docker container: `sglang_bbuf`
- Target GPU: NVIDIA B200, **GPU id 6** (`REMOTE_GPU_ID=6`, pinned via `CUDA_VISIBLE_DEVICES=6` for every build / correctness / benchmark / floor command)
- Remote task workspace: `/home/sglang-omni/bbuf/kda/sgl_kernel_build_tree_kernel_efficient/`
- Task source commit (this worktree base): `47114cdca5ee630ce599c91abce17b8fb95c3a4d`
- Upstream baseline commit: `7e6587c94a1d0305815a14067c5d3cc02a9b0f36`

## GPU idleness (performance data valid only when idle)
GPU 6 was idle before and after the measured run.

### Before
```
GPU 6, NVIDIA B200, util 0 %, mem 0 MiB / 183359 MiB
```

### After
```
GPU 6, NVIDIA B200, util 0 %, mem 0 MiB / 183359 MiB
```

## Software versions
- PyTorch: 2.11.0+cu130
- CUDA / nvcc: 13.0
- GPU architecture: `sm_100` (Blackwell B200)
- Build: torch `cpp_extension.load`, single combined build of baseline + candidate + binding, `-O3` both sides, no fast-math.

## Commands run (exact, all pinned to GPU 6)
```
# build (JIT, untimed)
CUDA_VISIBLE_DEVICES=6 python -c "import sys; sys.path.insert(0,'bench'); import build_ext; build_ext.get_ext()"
# correctness  -> ran 83 cases; 0 failures; CORRECTNESS: PASS
CUDA_VISIBLE_DEVICES=6 python bench/correctness.py
# benchmark    -> 15/15 PASSED; production geomean 1.144x
CUDA_VISIBLE_DEVICES=6 python bench/benchmark.py --device cuda:0 --warmup-runs 10 --num-trials 7 \
  --inner-iterations-min 1 --inner-iterations-max 4096 --target-sample-us 1000 --out bench/results.jsonl
# empty-kernel launch floor (build_tree_noop), same CUDA-event + amplification method
CUDA_VISIBLE_DEVICES=6 python bench/floor_probe.py
```

## Notes
- The standard cross-subprocess harness is clock-noisy for these sub-5µs kernels:
  the bs6/bs7/bs9 subprocesses ran ~5.3µs for BOTH sides (a low-clock state) vs
  ~4.1µs elsewhere, which is why bs7/bs9 show official speedup < 1.0. A controlled
  same-process probe (`bench/floor_probe.py`, 31 trials) shows the candidate
  clean-faster (non-overlapping p10/p90) for bs 2–10, geomean 1.021×, no
  regression. See `docs/results.md`.
- One code fix during bring-up: the non-contiguous fallback test built
  `verified_seq_len` from a `torch.empty((bs,2))` whose 2nd column was
  uninitialized; the contiguous-assuming baseline read that garbage → illegal
  memory access. Fixed in `bench/adapter.py` and `bench/correctness.py` to set
  both columns (uniform split) so the read stays in-bounds while the view is still
  non-contiguous (candidate still falls back). Kernels / ABI / workloads built and
  passed first try.
- Raw build log kept local/unstaged (`/tmp/build_tree_build_local.log` and the
  remote workspace `build_tree_build.log`); `bench/results.jsonl` copied back
  locally (gitignored).
