"""Differential probe: confirm the TVM-FFI ABI is callable, the baseline runs on GPU, matches the
independent naive oracle, and the candidate equals the baseline (naive) or is a valid top-k (radix).
Covers decode/prefill, naive/radix, contiguous/ties, NON-LINEAR (permuted) page tables, and a
row_starts (ragged-prefill) case. Resolves the tvm-ffi Module function-access pattern adapter.py uses.

This is a lightweight bring-up DIAGNOSTIC. The exhaustive frozen captured contract (all 251 rows,
incl. the 4 large-prefill row_starts tensor variants) is authoritatively gated by bench/correctness.py
(matched_ratio==1.0); this script mirrors its naive-oracle / valid-top-k criteria on a few cases.
Run on the B200: CUDA_VISIBLE_DEVICES=1 python3 solution/_probe.py
"""
import importlib.util as u
import torch

s = u.spec_from_file_location("b", "solution/build.py")
m = u.module_from_spec(s)
s.loader.exec_module(m)
mod = m.load_abi()
print("module type:", type(mod).__module__ + "." + type(mod).__name__)


def resolve(name):
    if hasattr(mod, name):
        return getattr(mod, name), "getattr"
    try:
        return mod[name], "getitem"
    except Exception:
        pass
    for g in ("get_function", "get_func"):
        if hasattr(mod, g):
            try:
                return getattr(mod, g)(name), g
            except Exception:
                pass
    raise RuntimeError("cannot resolve " + name)


base, how = resolve("fast_topk_transform_fused_baseline")
cand, _ = resolve("fast_topk_transform_fused_candidate")
print("ACCESS_PATTERN:", how)

dev = "cuda:0"  # pinned via CUDA_VISIBLE_DEVICES=1
topk = 2048


def seq_per_row(B, cu, is_decode):
    if is_decode:
        return list(range(B))
    cul = cu.tolist()
    return [next(si for si in range(len(cul) - 1) if cul[si] <= b < cul[si + 1]) for b in range(B)]


def case(B, N, S, M, dist="random", lengths_mode="full", page_table_mode="arange", row_starts_kind="none"):
    torch.manual_seed(0)
    g = torch.Generator(device=dev).manual_seed(0)
    if dist == "ties":
        score = torch.randint(0, 8, (B, N), device=dev, generator=g).float()
    else:
        score = torch.randn(B, N, device=dev, generator=g)
    L = min(N, M)
    if lengths_mode == "half":
        L = max(0, L // 2)
    lengths = torch.full((B,), L, dtype=torch.int32, device=dev)
    pt = torch.arange(S * M, dtype=torch.int32, device=dev).reshape(S, M)
    if page_table_mode == "permuted":  # non-linear: exercises the real-page-table inversion, not out-pt[s,0]
        pt = torch.stack([pt[i][torch.randperm(M, generator=g, device=dev)] for i in range(S)])
    if S == B:
        cu = torch.arange(S + 1, dtype=torch.int32, device=dev)
    else:
        base_, rem = divmod(B, S)
        acc = [0]
        for i in range(S):
            acc.append(acc[-1] + base_ + (1 if i < rem else 0))
        cu = torch.tensor(acc, dtype=torch.int32, device=dev)
    row_starts = None
    if row_starts_kind == "tensor":  # ragged prefill: row_start[b] + L <= N keeps the score window in bounds
        maxstart = max(0, N - L)
        rs = (torch.arange(B, device=dev, dtype=torch.int64) * 7) % (maxstart + 1) if maxstart else torch.zeros(B, device=dev, dtype=torch.int64)
        row_starts = rs.to(torch.int32)
    return score, lengths, pt, cu, L, row_starts


def naive_oracle(pt, lengths, cu, B, is_decode):
    """Naive path: dst[b,i] = (i<length) ? page_table[seq][i] : -1 (uses the ACTUAL page table, so it
    handles permuted rows; the naive path does not read score, so row_starts does not change it)."""
    out = torch.full((B, topk), -1, dtype=torch.int32, device=dev)
    seq = seq_per_row(B, cu, is_decode)
    for b in range(B):
        Lb = min(int(lengths[b]), topk)
        if Lb > 0:
            out[b, :Lb] = pt[seq[b], :Lb]
    return out


def valid_topk(out, score, lengths, pt, cu, B, is_decode, row_starts=None):
    """Radix valid-top-k (order/tie tolerant), mirroring bench/correctness.py validate_topk:
    invert each output page id through the ACTUAL page_table[seq] row (per-sequence scatter inverse,
    so it works for permuted tables), require positions distinct & in [0,length) & count==topk, and
    the selected score multiset (honoring the row_start score-window shift) == true top-k."""
    seq = seq_per_row(B, cu, is_decode)
    S, M = int(pt.shape[0]), int(pt.shape[1])
    maxv = int(pt.max().item()) + 1
    inv = torch.full((S, maxv), -1, dtype=torch.long, device=dev)
    cols = torch.arange(M, device=dev, dtype=torch.long)
    for si in range(S):
        inv[si, pt[si].long()] = cols
    for b in range(B):
        L = int(lengths[b])
        si = seq[b]
        rs = int(row_starts[b]) if row_starts is not None else 0
        ent = out[b].to(torch.int64)
        if not bool(((ent >= 0) & (ent < maxv)).all()):
            return False
        pos = inv[si, ent]
        if not bool(((pos >= 0) & (pos < L)).all()):
            return False
        if int(torch.unique(pos).numel()) != topk:
            return False
        sel = score[b, rs + pos].float()
        true_top = torch.topk(score[b, rs:rs + L].float(), topk).values
        if not torch.equal(torch.sort(sel).values, torch.sort(true_top).values):
            return False
    return True


def run(name, B, N, S, M, page_table_mode="arange", row_starts_kind="none", **kw):
    score, lengths, pt, cu, L, row_starts = case(
        B, N, S, M, page_table_mode=page_table_mode, row_starts_kind=row_starts_kind, **kw)
    is_decode = (S == B) and (row_starts is None)  # baseline decode branch: row_starts None && S==B
    dst = torch.full((B, topk), -17, dtype=torch.int32, device=dev)
    base(score, lengths, pt, cu, topk, row_starts, dst)
    torch.cuda.synchronize()
    dst_c = torch.full((B, topk), -17, dtype=torch.int32, device=dev)
    cand(score, lengths, pt, cu, topk, row_starts, dst_c)
    torch.cuda.synchronize()
    tags = f"pt={page_table_mode} rs={row_starts_kind}"
    if L <= topk:  # naive path: deterministic -> exact candidate==baseline + oracle
        cc = torch.equal(dst, dst_c)
        orc = naive_oracle(pt, lengths, cu, B, is_decode)
        ok = torch.equal(dst, orc)
        print(f"[{name}] B={B} N={N} S={S} M={M} L={L} {tags} candidate==baseline={cc} baseline==oracle={ok}")
        return cc and ok
    # radix path: output order is race-nondeterministic -> validate each output as a valid top-k
    okb = valid_topk(dst, score, lengths, pt, cu, B, is_decode, row_starts)
    okc = valid_topk(dst_c, score, lengths, pt, cu, B, is_decode, row_starts)
    print(f"[{name}] B={B} N={N} S={S} M={M} L={L} {tags} baseline_valid_topk={okb} "
          f"candidate_valid_topk={okc} (exact_order_match={torch.equal(dst, dst_c)}, expected False)")
    return okb and okc


def fallback_regression():
    """Out-of-contract metadata must take the baseline FALLBACK, not the native kernel. Here `score`
    has a batch that mismatches `dst` (the baseline defines B = score.size(0) and rejects
    dst.size(0) != B), so a correct candidate falls back and raises exactly like the baseline. Without
    the `score.size(0) == batch` dispatch guard the native bucket would still fire (the kernel never
    reads score) and silently return a wrong-contract output -> this is the regression guard."""
    B, N, S, M = 8, 64, 8, 64
    score, lengths, pt, cu, L, _ = case(B, N, S, M)
    bad_score = torch.randn(B + 1, N, device=dev)  # mismatched batch (B+1 vs dst batch B)
    dst = torch.full((B, topk), -17, dtype=torch.int32, device=dev)

    def raises(fn, sc):
        try:
            fn(sc, lengths, pt, cu, topk, None, dst)
            torch.cuda.synchronize()
            return False
        except Exception:
            return True

    cr = raises(cand, bad_score)
    br = raises(base, bad_score)
    ok_ctrl = not raises(cand, score)  # control: the valid-batch score must NOT raise (native path runs)
    print(f"[fallback_regression] score_batch_mismatch candidate_raised={cr} baseline_raised={br} "
          f"control_native_ok={ok_ctrl} (all expected True)")
    return cr and br and ok_ctrl


results = []
results.append(run("decode_small", 2, 64, 2, 40))
results.append(run("decode_contig_eqtopk", 8, 2048, 8, 2048))
results.append(run("radix_gt_topk", 8, 2112, 8, 2112))
results.append(run("ties", 8, 256, 8, 256, dist="ties"))
results.append(run("prefill", 16, 448, 4, 409))
results.append(run("permuted_naive", 4, 128, 4, 128, page_table_mode="permuted"))
results.append(run("permuted_radix", 8, 2112, 8, 2112, page_table_mode="permuted"))
results.append(run("row_starts_radix", 8, 2304, 8, 2176, row_starts_kind="tensor"))
results.append(fallback_regression())  # out-of-contract input must fall back, not run native
import sys
print("PROBE_OK" if all(results) else "PROBE_FAIL")
sys.exit(0 if all(results) else 1)
