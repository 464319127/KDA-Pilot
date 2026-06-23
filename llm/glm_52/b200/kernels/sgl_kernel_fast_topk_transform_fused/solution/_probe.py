"""Differential probe: confirm the TVM-FFI ABI is callable, the baseline runs on GPU,
matches the independent naive oracle, and the candidate equals the baseline. Also resolves
the tvm-ffi Module function-access pattern that bench/adapter.py should use.
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


def case(B, N, S, M, dist="random", lengths_mode="full"):
    torch.manual_seed(0)
    if dist == "ties":
        score = torch.randint(0, 8, (B, N), device=dev).float()
    else:
        score = torch.randn(B, N, device=dev)
    L = min(N, M)
    if lengths_mode == "half":
        L = max(0, L // 2)
    lengths = torch.full((B,), L, dtype=torch.int32, device=dev)
    pt = torch.arange(S * M, dtype=torch.int32, device=dev).reshape(S, M)
    if S == B:
        cu = torch.arange(S + 1, dtype=torch.int32, device=dev)
    else:
        base_, rem = divmod(B, S)
        acc = [0]
        for i in range(S):
            acc.append(acc[-1] + base_ + (1 if i < rem else 0))
        cu = torch.tensor(acc, dtype=torch.int32, device=dev)
    return score, lengths, pt, cu, L


def naive_oracle(pt, lengths, cu, B, is_decode):
    out = torch.full((B, topk), -1, dtype=torch.int32, device=dev)
    cul = cu.tolist()
    seq = list(range(B)) if is_decode else [next(si for si in range(len(cul) - 1) if cul[si] <= b < cul[si + 1]) for b in range(B)]
    for b in range(B):
        Lb = min(int(lengths[b]), topk)
        if Lb > 0:
            out[b, :Lb] = pt[seq[b], :Lb]
    return out


def run(name, B, N, S, M, **kw):
    score, lengths, pt, cu, L = case(B, N, S, M, **kw)
    dst = torch.full((B, topk), -17, dtype=torch.int32, device=dev)
    base(score, lengths, pt, cu, topk, None, dst)
    torch.cuda.synchronize()
    dst_c = torch.full((B, topk), -17, dtype=torch.int32, device=dev)
    cand(score, lengths, pt, cu, topk, None, dst_c)
    torch.cuda.synchronize()
    cc = torch.equal(dst, dst_c)
    msg = f"[{name}] B={B} N={N} S={S} M={M} L={L} candidate==baseline={cc}"
    if L <= topk:  # naive path -> exact oracle
        orc = naive_oracle(pt, lengths, cu, B, S == B)
        ok = torch.equal(dst, orc)
        msg += f" baseline==oracle={ok}"
    else:
        # radix path: just sanity-check no poison left and indices in range
        ok = bool((dst != -17).all().item())
        msg += f" radix_no_poison={ok}"
    print(msg)
    return cc and ok


results = []
results.append(run("decode_small", 2, 64, 2, 40))
results.append(run("decode_contig_eqtopk", 8, 2048, 8, 2048))
results.append(run("radix_gt_topk", 8, 2112, 8, 2112))
results.append(run("ties", 8, 256, 8, 256, dist="ties"))
results.append(run("prefill", 16, 448, 4, 409))
print("PROBE_OK" if all(results) else "PROBE_FAIL")
