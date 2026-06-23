# Remote Run Log — fast_topk_transform_fused (B200)

All GPU work runs on the task's pinned host/GPU. No fabricated evidence; entries are appended per session.

## Session — Round 2 (ABI build + differential probe)

- Host: `ion-b200` → `innomatrix-us-adc-smb200-0003`, user `sglang-omni`, container `sglang_bbuf` (Up 8 days).
- Toolchain: torch `2.11.0+cu130`, CUDA `13.0`, nvcc `13.0.88`, `tvm_ffi 0.1.9`, flashinfer present.
- Task workspace on remote: `/home/sglang-omni/bbuf/kda_topk` (synced from the local kernel folder via tar; `.humanize`/`__pycache__` excluded).
- Pinned GPU: id 1 (`REMOTE_GPU_ID=1` → `CUDA_VISIBLE_DEVICES=1`).

### GPU state (nvidia-smi, before work)
```
0, NVIDIA B200, util 5%,  mem 53638/183359 MiB
1, NVIDIA B200, util 5%,  mem 41472/183359 MiB   <-- pinned GPU 1: NOT idle (other tenant ~41 GB)
2, NVIDIA B200, util 8%,  mem 85054/183359 MiB
3..7 idle (0 util, 0 mem except 5=156764)
```
GPU 1 is **not idle** this session. Correctness/probe (exact-match, not timing-sensitive) ran on GPU 1's ~142 GB free memory — valid for correctness. **Timing/benchmark (AC-7) is deferred** until GPU 1 is idle (verified before+after), per the task policy; no benchmark numbers were taken this session.

### Commands
```
# build the TVM-FFI ABI (cached in solution/)
cd /home/sglang-omni/bbuf/kda_topk && TORCH_CUDA_ARCH_LIST=10.0 python3 solution/build.py
# -> built+loaded topk_transform_abi : OK

# differential probe (one output + naive-oracle correctness + candidate==baseline)
CUDA_VISIBLE_DEVICES=1 python3 solution/_probe.py
```

### Results
- **ABI build:** SUCCESS. `tvm_ffi.cpp.load_inline` compiled baseline `topk.cu` + candidate + TVM-FFI binding into one `topk_transform_abi` module (sm_100, `-std=c++17`, no fast-math; torch include/lib linked for the ATen-based baseline). Module type `tvm_ffi.module.Module`; functions accessed via attribute (`mod.fast_topk_transform_fused_baseline(...)`).
- **Output count:** confirmed ONE `(B, topk)` int32 output (destination-passing signature; the op writes exactly `dst_out`). Final hardware confirmation of the static finding in `docs/baseline_source.md`.
- **Naive path (`length <= topk`)** — `baseline == naive oracle` AND `candidate == baseline`, exact, on: decode (B=2,N=64,M=40), N==topk (B=8,N=2048), ties (B=8,N=256), prefill (B=16,N=448,S=4). This is the dominant production regime (3674/4246 captured calls have `N < topk`).
- **Radix path (`length > topk`, e.g. B=8,N=2112)** — `candidate != baseline` **even though the candidate forwards to the identical baseline code**. The baseline `fast_topk_cuda_tl` is therefore **non-deterministic**: it quantizes each score to an 8-bit key (`convert_to_uint8`), so many distinct float scores collide into the boundary histogram bucket, and the fill of the last `k` slots from that bucket is thread/race-ordered. Two runs on identical inputs select different (equally-valid) indices.

### Consequence (carried to next round)
AC-4's literal "exact-match selected indices + baseline-identical tie-break" is **infeasible for the radix path** because the baseline itself is non-deterministic there. The correctness criterion must split by regime: naive path = exact-match (works now); radix path = a tie-tolerant valid-top-k criterion (selected indices' 8-bit keys all ≥ the threshold key; per-bucket counts and the transform/pad match), comparing against the baseline's 8-bit-key selection semantics rather than a single non-deterministic run.
