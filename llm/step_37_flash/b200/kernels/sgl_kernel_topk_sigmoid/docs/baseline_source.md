# Baseline Source Provenance — `sgl_kernel.topk_sigmoid`

## Upstream resolution

- Repository: https://github.com/sgl-project/sglang
- Branch: `main`
- Resolved commit SHA: `5e6d7c1615a95dc5f98e69b4b18af0ae160b10b8`
- Resolution method: `git ls-remote https://github.com/sgl-project/sglang.git refs/heads/main`
- Resolution date: 2026-06-24 (recovered at baseline-recovery time per the task contract)
- Python interface recovered: `sgl_kernel.topk_sigmoid`

## Torch op schema (authoritative I/O contract)

From `sgl-kernel/csrc/common_extension.cc`:

```
topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output,
             bool renormalize, Tensor? correction_bias) -> ()
```

- Return type is `-> ()` → the op returns **None (void)**.
- `Tensor!` marks `topk_weights` (arg0) and `topk_indices` (arg1) as **mutable / written in place**.
- `gating_output` (arg2) has **no `!` → read-only; the op must not mutate it**.
- `correction_bias` (arg4) is `Tensor?` (optional); in all captured variants it is provided as `[288] float32`.

Python wrapper (`sgl-kernel/python/sgl_kernel/moe.py`): `topk_sigmoid(topk_weights, topk_ids,
gating_output, renormalize=False, correction_bias=None) -> None`; it forwards to
`torch.ops.sgl_kernel.topk_sigmoid.default(...)` and returns `None`.

### Reconciliation with the captured evidence

`docs/evidence.json` logged each call's result as both `return=None` and a 3-tuple
`([N,8] f32, [N,8] i32, [N,288] f32)`. The op truly returns `None`; the "tuple" is a logging
artifact — the capture tool dumped the relevant tensors after the call (arg0/arg1 written in
place, arg2 unchanged). **Conclusion: outputs are the in-place writes to arg0 (topk_weights)
and arg1 (topk_indices); arg2 (gating_output) is not mutated.** This will be re-confirmed by a
runtime data_ptr/poison probe when the ABI is built on B200.

## Files inspected upstream (at the resolved commit)

- `sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.cu` — the kernel + host launcher (the baseline).
- `sgl-kernel/python/sgl_kernel/moe.py` — Python wrapper (`topk_sigmoid`).
- `sgl-kernel/csrc/common_extension.cc` — torch library op schema/registration.
- `sgl-kernel/tests/test_moe_topk_sigmoid.py` — upstream tests (reference oracle, see Semantics).
- `sgl-kernel/include/utils.h` — source of `WARP_SIZE`, `SGLANG_SHFL_XOR_SYNC_WIDTH`.

## Files copied into `baseline/`

| File | Origin | Status | sha256 (copied) |
|------|--------|--------|------------------|
| `baseline/topk_sigmoid_baseline.cu` | `sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.cu` @ resolved commit | verbatim, no edits | `4b8eadf84561c8b71c31893508caacc2c132a15510c725385616775a59e16ce3` |
| `baseline/utils.h` | minimal shim of `sgl-kernel/include/utils.h` | local shim (only `WARP_SIZE`, `SGLANG_SHFL_XOR_SYNC[_WIDTH]`, CUDA branch copied verbatim) | n/a |

Local edits: the vendored `.cu` is unchanged. `baseline/utils.h` is a minimal shim (not the full
upstream header) providing only the two symbols the kernel references, both on the unused
power-of-two path. The local ABI wrapper (added separately) calls the vendored `topk_sigmoid(...)`
C++ function directly; the upstream torch-library registration is NOT used at runtime (no live
SGLang import).

## Recovered semantics (the captured 288-expert case)

For `num_experts = 288` the host launcher takes `needs_workspace = !is_pow_2 || num_experts > 256`
→ **true** (288 is not a power of two). It therefore uses the **two-launch workspace path**:

1. `moeSigmoid` (one block per token, TPB=256) writes `sigmoid(gating[r,e]) + correction_bias[e]`
   (the **biased selection score**) into a `torch::empty({N*288}, float32)` workspace.
2. `moeTopK` (one block per token) iterates `k=8` times, each time taking a `cub::BlockReduce`
   `cub::ArgMax` over the 288 scores, skipping already-selected experts. For each pick it writes
   `weight = score - correction_bias[expert]` (the **unbiased** sigmoid value) and the selected
   expert id; finally, with `renormalize=True`, it divides the 8 weights by their sum.

Net algorithm (`renormalize=True`, bias present — the captured configuration):

1. `score[e] = sigmoid(gating[r,e]) + bias[e]`  (fp32)
2. select top-8 experts by `score` (descending); **tie-break: lower expert index wins**
   (`cub::ArgMax` keeps the smaller key on equal value; the fast path uses the same rule explicitly).
3. output `weight = score - bias = unbiased sigmoid(gating[r,e])`.
4. `weight[i] /= sum_{j in top8} weight[j]`.
5. `topk_indices[r,i]`, `topk_weights[r,i]` are in selection order (descending biased score).

Reference oracle (matches upstream `test_topk_sigmoid_renormalize_correction_bias`):

```python
sigmoid = torch.sigmoid(gating_output)                       # fp32
scores  = sigmoid + correction_bias.unsqueeze(0)             # biased selection score
_, idx  = torch.topk(scores, k=8, dim=-1)                    # select by biased score
w       = sigmoid.gather(1, idx)                             # unbiased weights
w       = w / w.sum(dim=-1, keepdim=True)                    # renormalize (renormalize=True)
```

Note: `torch.topk` tie-breaking is unstable and may disagree with the kernel on exact ties; the
**recovered baseline is the authoritative oracle** for constructed tie rows (exact id match).

## Optimization lever (for the candidate, not the baseline)

288 experts force the baseline onto the slow workspace path: **two kernel launches plus a
`torch::empty` workspace allocation plus a full [N,288] fp32 global round-trip**. A single fused
native-CUDA kernel that computes sigmoid + biased selection + top-8 + unbiased weights +
renormalization in one launch (no workspace) is the candidate direction. For the dominant `N=1`
decode case the op is tiny and likely launch/SFU/reduction-latency bound; a controlled
empty-kernel floor probe at the same grid will bound how much launch/allocation overhead is
recoverable.
