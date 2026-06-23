# Baseline Source Provenance — `sgl_kernel.verify_tree_greedy`

## Upstream resolution

| Field | Value |
|-------|-------|
| Repository URL | https://github.com/sgl-project/sglang |
| Branch | `main` |
| Resolved commit SHA | `7e6587c94a1d0305815a14067c5d3cc02a9b0f36` |
| Resolution time (UTC) | 2026-06-23T05:49:27Z |
| Resolution method | `git ls-remote refs/heads/main` then blobless sparse checkout of `sgl-kernel/`; `git rev-parse HEAD` confirmed the same SHA (main did not move between resolve and checkout). |

## Copied files

All copies are verbatim from the commit above. No local edits were made to the copied sources.

| Local path (under `baseline/`) | Upstream path | Purpose |
|--------------------------------|---------------|---------|
| `eagle_utils.cu` | `sgl-kernel/csrc/speculative/eagle_utils.cu` | Contains the `VerifyTreeGreedy<>` CUDA kernel (defined at the `verify_tree_greedy` host function) — the baseline kernel for this task. (The file also contains other EAGLE tree-build kernels that are not used by this task.) |
| `speculative.py` | `sgl-kernel/python/sgl_kernel/speculative.py` | Python wrapper exposing `sgl_kernel.verify_tree_greedy` (reference for argument order). |
| `upstream_test_eagle_utils.py` | `sgl-kernel/tests/speculative/test_eagle_utils.py` | Upstream reference test with hand-checked expected outputs — used as the CPU oracle anchor. |

## Public interface (Python)

`sgl_kernel.verify_tree_greedy` (from `speculative.py`), in-place, returns `None`:

```
verify_tree_greedy(
    predicts,            # mutable OUTPUT, (num_tokens,)       int32
    accept_index,        # mutable OUTPUT, (bs, num_spec_step) int32
    accept_token_num,    # mutable OUTPUT, (bs,)               int32
    candidates,          #          INPUT, (bs, num_draft_tokens) int64
    retrive_index,       #          INPUT, (bs, num_draft_tokens) int64
    retrive_next_token,  #          INPUT, (bs, num_draft_tokens) int64
    retrive_next_sibling,#          INPUT, (bs, num_draft_tokens) int64
    target_predict,      #          INPUT, (bs, num_draft_tokens) int64
)
```

It dispatches to `torch.ops.sgl_kernel.verify_tree_greedy.default(...)` (no stream argument; the C++ side uses the current CUDA stream).

## C++ host signature and registration

- Declaration: `sgl-kernel/include/sgl_kernel_ops.h` (`void verify_tree_greedy(at::Tensor predicts, at::Tensor accept_index, at::Tensor accept_token_num, at::Tensor candidates, at::Tensor retrive_index, at::Tensor retrive_next_token, at::Tensor retrive_next_sibling, at::Tensor target_predict);`).
- Definition: `sgl-kernel/csrc/speculative/eagle_utils.cu` (host `verify_tree_greedy`, kernel `VerifyTreeGreedy<int32_t, int64_t>`).
- Registration schema: `sgl-kernel/csrc/common_extension.cc`
  ```
  verify_tree_greedy(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num,
                     Tensor candidates, Tensor retrive_index, Tensor retrive_next_token,
                     Tensor retrive_next_sibling, Tensor target_predict) -> ()
  ```
  (`Tensor!` marks the three mutable outputs; registered for `torch::kCUDA`.)

## Launch configuration and dtype contract

- Launch: `dim3 grid(batch_size); dim3 block(1);` — **one block, one thread per request**, on `at::cuda::getCurrentCUDAStream()`. No shared memory.
- Template instantiation: `VerifyTreeGreedy<int32_t /*outputs*/, int64_t /*inputs*/>`.
- Host validates (and the standalone baseline/candidate must honor): `predicts/accept_index/accept_token_num` are `int32`; `candidates/retrive_index/retrive_next_token/retrive_next_sibling/target_predict` are `int64`; the dim checks `predicts` 1-D, `accept_index` 2-D `(bs, num_spec_step)`, the five inputs 2-D `(bs, num_draft_tokens)`.

## Recovered algorithm (exact, per request `bx`)

`num_draft_tokens = candidates.size(1)`, `num_spec_step = accept_index.size(1)`. Index conventions: `retrive_index[bx*nd + col]` is the **GLOBAL / flattened** node index; `cur_index` is a **ROW-LOCAL column** index; `candidates`, `retrive_next_token`, `retrive_next_sibling` are indexed row-locally by `bx*nd + col`; `target_predict` and `predicts` are indexed by the **GLOBAL** index.

```
last_accepted = retrive_index[bx*nd + 0]          # global index of root
accept_index[bx*nss + 0] = last_accepted          # root recorded even on full reject
num_accepted = 0
cur = 0                                            # row-local column
for j in 1 .. num_spec_step-1:
    cur = retrive_next_token[bx*nd + cur]          # first child (row-local)
    while cur != -1:
        draft_index = retrive_index[bx*nd + cur]   # global index of this child
        draft_token = candidates[bx*nd + cur]      # row-local
        target_token = target_predict[last_accepted]   # GLOBAL index into target_predict
        if draft_token == target_token:
            predicts[last_accepted] = target_token      # write at parent's global slot
            num_accepted += 1
            accept_index[bx*nss + num_accepted] = draft_index
            last_accepted = draft_index
            break
        else:
            cur = retrive_next_sibling[bx*nd + cur]     # next sibling (row-local)
    if cur == -1: break
accept_token_num[bx] = num_accepted                # counts accepted DRAFT tokens (excludes root)
predicts[last_accepted] = target_predict[last_accepted]   # bonus token at last accepted slot
```

Output positions NOT on the accepted path are **left untouched** (predicts entries off the path; `accept_index[bx][num_accepted+1 ..]`). Exact-match correctness must therefore compare against poisoned buffers so untouched slots are validated too.

Verified by hand against `upstream_test_eagle_utils.py` (bs=2, nd=6, num_spec_step=4): reproduces `predicts == [3,-1,-1,4,5,18,11,-1,-1,-1,12,18]`, `accept_index == [[0,3,4,5],[6,10,11,-1]]`, `accept_token_num == [3,2]`, including a sibling hop on request 0 (child col 1 token=1 ≠ target 3 → sibling col 3 token=3 == target 3 → accept).

## Build dependencies (for the standalone baseline ABI in a later round)

`eagle_utils.cu` includes `<ATen/ATen.h>`, `<ATen/cuda/CUDAContext.h>`, and `pytorch_extension_utils.h`. The `CHECK_INPUT` / `CHECK_DIM` / `CHECK_EQ` macros it uses are defined in `sgl-kernel/include/utils.h` (pulled in via `pytorch_extension_utils.h`). The standalone baseline build will either vendor the minimal macro set or provide equivalents; this is handled when the local baseline ABI wrapper is added.

## Captured regime for this task

`num_draft_tokens = 2`, `num_spec_step = 2`, `bs ∈ {1..10}`, all tensors contiguous, no scalar kwargs. With `num_spec_step = 2` the captured rows accept at most one draft token, so deeper/wider synthetic trees (e.g. the upstream test's nd=6/spec=4 tree) are needed to exercise partial-accept and sibling traversal; those route to the baseline via fallback in the candidate.
