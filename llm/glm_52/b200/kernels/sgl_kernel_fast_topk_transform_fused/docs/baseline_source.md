# Baseline Source Provenance — `sgl_kernel.fast_topk_transform_fused`

## Upstream lineage
- Repository URL: https://github.com/sgl-project/sglang
- Branch: `main`
- Resolved commit SHA: `7e6587c94a1d0305815a14067c5d3cc02a9b0f36`
- Resolution time (UTC): `2026-06-23T05:53:16Z`
- Resolution method: `git ls-remote https://github.com/sgl-project/sglang.git refs/heads/main` to resolve the latest `main` tip, then `curl https://raw.githubusercontent.com/sgl-project/sglang/<SHA>/<path>` to copy each file at that exact commit. The shared local checkout at `/Users/bbuf/工作目录/Common/sglang` was used read-only to locate the files; it was on a drifted local branch (`bbuf/dmd-fastpath-speed-policy`, not main) and was NOT used as the source of truth.
- Local edits to copied files: none (verbatim copies at the resolved SHA).

## Copied files (under `baseline/`, upstream layout preserved)
| File | Role | sha256 |
|------|------|--------|
| `sgl-kernel/csrc/elementwise/topk.cu` | CUDA kernels + `fast_topk_transform_interface` (the C++ op impl) | `f899cb9d2db331f2315e84f89c48716863870082f76ae65428ae4d865b0205d7` |
| `sgl-kernel/python/sgl_kernel/top_k.py` | Python wrapper `fast_topk_transform_fused(...)` | `567c1c5725fad322b6e7b2d54b216a48740c9c66b58af6ed8f45a944c67fda56` |
| `sgl-kernel/csrc/common_extension.cc` | Torch-library op registration (reference) | `7cefc96373c977f5220079097412a0eb6228d180744963eed0c4d43e83b1a0a1` |
| `sgl-kernel/python/sgl_kernel/__init__.py` | Public export of `fast_topk_transform_fused` (reference) | `82119a6bef1cae7195162b580a1c4e28075bc32ba125345470c291d46a04cc54` |

`topk.cu` includes only external headers (ATen / c10 / CUDA / libstdc++), so the kernel source is self-contained — no local sgl-kernel headers were required. The build wiring (exposing `fast_topk_transform_interface` through the task-local ABI) is recovered/built in the next round on the remote B200, where the PyTorch/CUDA toolchain is available.

## Recovered semantics (from `topk.cu` + `top_k.py`, to be confirmed by differential probes at build time)
This is the DeepSeek-V3.2-style sparse-attention top-k indexer (`topk == 2048` asserted in the wrapper; GLM-5.2 reuses it). The op is destination-passing: `fast_topk_transform_fused(Tensor score, Tensor lengths, Tensor dst_page_table, Tensor src_page_table, Tensor cu_seqlens_q, Tensor? row_starts)`, impl `fast_topk_transform_interface` (`topk.cu` line ~456).

- Per-row selection picks the top-`topk` positions of `score[row]` over the row's valid range, then **transforms** each selected raw position `p` into a page-table entry `src_page_table[row, p]`, writing it to `dst_page_table[row, :]`.
- `naive_topk_transform` path (length ≤ topk): selects ALL valid positions `[0, length)` in order (no real selection) and **pads the tail `[length, topk)` with `-1`**. This matches the captured `N < topk` majority (3674/4246 calls) — for most production calls this is a copy/pad/transform, not a true top-k.
- `fast_topk_cuda_tl` path (length > topk): a TileLang-derived radix/histogram top-k selector — an 8-bit coarse histogram then a 32-bit `convert_to_uint32` refinement of the boundary bin (topk.cu lines ~184-251). It selects the TRUE top-k by full float precision, but the output ORDER (and the choice among exactly-equal-value ties) is race-nondeterministic (atomicAdd fill order). CONSEQUENCE (confirmed on B200, Round 2 — see docs/run_log.md): a candidate cannot be required to byte-match a single baseline run on this path. Correctness for `length>topk` is a VALID-top-k check (selected positions distinct and in `[0,length)`, count `topk`, and the multiset of their scores equal to the true top-k scores), tolerant of order and equal-value ties. Exact byte-match remains required for the deterministic `length<=topk` naive path.
- Dispatch: `topk_transform_decode_kernel` vs `topk_transform_prefill_kernel` (prefill uses `cu_seqlens_q` to map tokens→sequences; `prefill_bs` small). `static_assert(TopK / kThreadsPerBlock == 2)` → 1024 threads emit 2048 outputs (2 per thread).
- `row_starts`: in the GLM-5.2 capture, **243 of 247 variants have `row_starts=None`**, but **4 large-prefill variants (88 calls)** pass a `(B,)` int32 `row_starts` tensor (variants 127/129/222/223: score `[2334,2334]`,`[3080,3080]`,`[2207,5151]`,`[2366,4606]`, all `length>topk`). When non-`None`, the radix path reads scores from the ragged sub-window `score[b, row_starts[b] : row_starts[b]+lengths[b]]` (`fast_topk_cuda_tl` uses `input[idx + row_start]`); the selected relative positions still index `src_page_table[seq]`. `bench/workloads.json` keys `row_starts_kind` and the adapter synthesizes a valid `row_starts` (row_start+length ≤ N) for those rows.

### OUTPUT-COUNT CONTRACT — resolved to ONE output (static evidence; remote probe = final confirmation)
The interface returns **one** `(B, topk)` int32 tensor per call (`dst_page_table`). Evidence:
1. C++ `fast_topk_transform_interface` writes a single `dst_page_table` (destination-passing); there is no second output buffer.
2. `top_k.py` allocates one `dst_page_table` and `return`s it.
3. The caller `python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py` consumes a single return value: `return fast_topk_transform_fused(...)`.
4. In `docs/evidence.json` the per-variant `result.tensors` list has two entries, but **56 variants have entries with DIFFERING row counts** (e.g. variant[6]: score `B=3`, results `(3,2048)` and `(6,2048)`; the raw shows two separate `Output:` blocks). Since each call's output is `(score_B, 2048)`, a `(6,2048)` entry on a `B=3` variant must come from a *different call* — so `result.tensors` **aggregates the single-tensor returns of two sampled calls** of the variant, not two outputs of one call.

Therefore the harness allocates/verifies **one** output. (This corrects the earlier "logger duplication" phrasing: the mechanism is multi-call aggregation in the capture record, not duplication of one buffer.) The remote differential probe in the build round is the final on-hardware confirmation: run the recovered op once and assert it returns exactly one `(B, topk)` int32 tensor.
