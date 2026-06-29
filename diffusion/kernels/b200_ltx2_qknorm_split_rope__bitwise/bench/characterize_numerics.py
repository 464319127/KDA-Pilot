"""Numerics characterization probe for the LTX2 qknorm + split-RoPE bit-exact task.

Pins, on THIS exact CUDA stack (B200), the bf16 rounding behavior the candidate
kernel must reproduce:
  (A) RMSNorm(H) affine-order + rstd form + reduction-order sensitivity, vs
      torch.nn.RMSNorm and (if present) torch.ops.aten._fused_rms_norm's rstd.
  (B) split-RoPE addcmul_ opmath (fp32 single-round vs fmaf vs bf16-rounded
      product) vs the eager apply_split_rotary_emb_eager reference.

No sglang import. Pure torch + the task-local eager reference. Runs on CUDA only.

Usage:
    python bench/characterize_numerics.py            # full characterization (needs CUDA)
"""

import platform
import sys
from pathlib import Path

import torch

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from baseline.ltx2_split_rope import apply_split_rotary_emb_eager  # noqa: E402

BF16 = torch.bfloat16
F32 = torch.float32
EPS = 1e-6


def i16(t):
    return t.contiguous().view(torch.int16)


def biteq(a, b):
    return torch.equal(i16(a), i16(b))


def n_mismatch(a, b):
    return int((i16(a) != i16(b)).sum().item())


# Boundary bf16 values that stress rounding midpoints.
_POOL = [
    0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0,
    1.0 + 2 ** -7, 1.0 - 2 ** -8, 0.5 + 2 ** -8, 2 ** -6, -2 ** -6,
    2 ** -8, -2 ** -8, 0.75, -0.75, 1.5, -1.5, 1.25, 0.125, 7.0,
]


def make_rms_inputs(device, H, n_rows, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    # Mix random rows with adversarial structured rows.
    x = torch.randn(n_rows, H, generator=g, device=device, dtype=F32).to(BF16)
    pool = torch.tensor(_POOL, dtype=F32, device=device).to(BF16)
    # adversarial rows: all-equal, single-outlier, alternating, boundary tiles
    k = min(n_rows, 8)
    idx = torch.arange(H, device=device)
    x[0] = pool[idx % pool.numel()]
    if n_rows > 1:
        x[1] = pool[(idx * 3) % pool.numel()]
    if n_rows > 2:
        x[2].fill_(1.0 + 2 ** -7)
    if n_rows > 3:
        x[3].fill_(2 ** -6); x[3, 0] = 7.0
    # weight: bf16 values near 1 plus boundary-targeted
    w = (1.0 + 0.1 * torch.randn(H, generator=g, device=device, dtype=F32)).to(BF16)
    w[: pool.numel()] = pool
    return x, w


def rms_variants(x, w):
    """Return dict name -> bf16 output, emulating per-element kernel math.

    All use fp32 statistics. Variants differ in (rstd form) x (affine order) x
    (reduction method). The reference is torch.nn.RMSNorm.
    """
    xf = x.to(F32)
    wf = w.to(F32)
    H = x.shape[-1]
    out = {}
    # reduction methods for mean-square
    ms_mean = xf.pow(2).mean(dim=-1, keepdim=True)         # torch reduction (pairwise)
    ms_sum = xf.pow(2).sum(dim=-1, keepdim=True) / H        # sum-then-divide
    for rname, ms in (("mean", ms_mean), ("sumdiv", ms_sum)):
        rstd_rsqrt = torch.rsqrt(ms + EPS)
        rstd_sqrt = 1.0 / torch.sqrt(ms + EPS)
        for sname, rstd in (("rsqrt", rstd_rsqrt), ("sqrtrcp", rstd_sqrt)):
            normed = xf * rstd
            # V1: weight multiply in fp32, single final cast
            out[f"V1_wfp32.{rname}.{sname}"] = (normed * wf).to(BF16)
            # V2: cast normed to bf16, then * weight (fp32), final cast
            out[f"V2_castnormed.{rname}.{sname}"] = (normed.to(BF16).to(F32) * wf).to(BF16)
    return out


def characterize_rmsnorm(device):
    print("\n===== (A) RMSNorm characterization =====")
    Hs = [256, 2048, 4096]
    seeds = list(range(8))
    rows = 192
    # which variants survive on ALL data
    survivors = None
    per_variant_mismatch = {}
    rstd_report = None
    for H in Hs:
        ref_mod = torch.nn.RMSNorm(H, eps=EPS, device=device, dtype=BF16)
        for seed in seeds:
            x, w = make_rms_inputs(device, H, rows, seed)
            with torch.no_grad():
                ref_mod.weight.copy_(w)
                y_ref = ref_mod(x)
            cands = rms_variants(x, w)
            ok_now = set()
            for name, y in cands.items():
                mm = n_mismatch(y, y_ref)
                per_variant_mismatch[name] = per_variant_mismatch.get(name, 0) + mm
                if mm == 0:
                    ok_now.add(name)
            survivors = ok_now if survivors is None else (survivors & ok_now)
            # rstd oracle (once)
            if rstd_report is None:
                rstd_report = probe_rstd_oracle(x, w, H, device)
    print("  Per-variant total mismatches across all H/seeds (0 == bit-exact everywhere):")
    for name in sorted(per_variant_mismatch):
        print(f"    {name:32s} mismatches={per_variant_mismatch[name]}")
    print(f"  SURVIVORS (bit-exact on every row/seed/H): {sorted(survivors) if survivors else 'NONE'}")
    # reduction sensitivity: did 'mean' vs 'sumdiv' ever change the matching variant's output?
    print(f"  rstd oracle: {rstd_report}")
    return survivors, per_variant_mismatch, rstd_report


def probe_rstd_oracle(x, w, H, device):
    """Compare candidate rstd (rsqrt of mean) against aten._fused_rms_norm rstd if available."""
    xf = x.to(F32)
    cand_rstd = torch.rsqrt(xf.pow(2).mean(dim=-1) + EPS)
    op = getattr(torch.ops.aten, "_fused_rms_norm", None)
    if op is None:
        return "aten._fused_rms_norm NOT available; rstd compared indirectly via output match"
    try:
        res = op.default(x, [H], w, EPS)
    except Exception as exc:  # noqa: BLE001
        return f"aten._fused_rms_norm present but call failed: {type(exc).__name__}: {exc}"
    if not isinstance(res, (tuple, list)) or len(res) < 2:
        return f"aten._fused_rms_norm returned {type(res)} (no separate rstd)"
    aten_rstd = res[1].to(F32).flatten()
    cand = cand_rstd.flatten()[: aten_rstd.numel()]
    exact = torch.equal(cand.view(torch.int32), aten_rstd.view(torch.int32))
    max_ulp = None
    if not exact:
        # crude ULP estimate via int32 distance of fp32 bit patterns
        a = cand.view(torch.int32).to(torch.int64)
        b = aten_rstd.view(torch.int32).to(torch.int64)
        max_ulp = int((a - b).abs().max().item())
    return f"aten._fused_rms_norm rstd: candidate-rsqrt-of-mean exact={exact} max_ulp={max_ulp}"


def make_rope_inputs(device, B, S, num_heads, head_dim, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    H = num_heads * head_dim
    r = head_dim // 2
    pool = torch.tensor(_POOL, dtype=F32, device=device).to(BF16)
    npool = pool.numel()
    t = torch.arange(S, device=device).view(S, 1)
    j = torch.arange(r, device=device).view(1, r)
    # x [B,S,H]: half boundary, half random
    x = torch.randn(B, S, H, generator=g, device=device, dtype=F32).to(BF16)
    x[..., :H // 2] = pool[(torch.arange(H // 2, device=device)) % npool]
    # cos/sin physical [B,S,heads,r] -> view [B,heads,S,r] (non-contig, last stride 1)
    cos_phys = pool[(t * 2 + j) % npool].view(1, S, 1, r).expand(B, S, num_heads, r).contiguous()
    sin_phys = pool[(t + j * 7) % npool].view(1, S, 1, r).expand(B, S, num_heads, r).contiguous()
    cos = cos_phys.transpose(1, 2)
    sin = sin_phys.transpose(1, 2)
    return x, cos, sin


def rope_variants(x, cos, sin, num_heads, head_dim):
    """Emulate per-element split-RoPE candidate variants; return dict name->bf16 [B,S,H]."""
    B, S, H = x.shape
    r = head_dim // 2
    xv = x.reshape(B, S, num_heads, head_dim).transpose(1, 2)  # [B,heads,S,head_dim]
    first = xv[..., :r]
    second = xv[..., r:]
    cosf = cos.to(F32)
    sinf = sin.to(F32)
    ff = first.to(F32)
    sf = second.to(F32)
    # visible first rounding: c = bf16(x*cos)
    c1 = (ff * cosf).to(BF16).to(F32)
    c2 = (sf * cosf).to(BF16).to(F32)
    out = {}

    def assemble(of, os_):
        o = torch.empty(B, num_heads, S, head_dim, dtype=BF16, device=x.device)
        o[..., :r] = of
        o[..., r:] = os_
        return o.transpose(1, 2).reshape(B, S, H).contiguous()

    # A1: fp32 mul then fp32 add, single bf16 cast
    out["A1_fp32_muladd"] = assemble(
        (c1 - sinf * sf).to(BF16), (c2 + sinf * ff).to(BF16))
    # A2: fmaf (single fp32 FMA): c +/- (sin*x) fused
    out["A2_fmaf"] = assemble(
        torch.addcmul(c1, -sinf, sf).to(BF16) if False else (c1 + (-sinf) * sf).to(BF16),
        (c2 + sinf * ff).to(BF16))
    # A2 proper fmaf via torch: emulate single-rounded fma is not directly exposed;
    # approximate with double precision to detect if eager used a single rounding.
    out["A2b_fma_dbl"] = assemble(
        (c1.double() - sinf.double() * sf.double()).to(F32).to(BF16),
        (c2.double() + sinf.double() * ff.double()).to(F32).to(BF16))
    # A3: round sin*x product to bf16 first, then fp32 add, cast
    out["A3_bf16_product"] = assemble(
        (c1 - (sinf * sf).to(BF16).to(F32)).to(BF16),
        (c2 + (sinf * ff).to(BF16).to(F32)).to(BF16))
    return out


def characterize_rope(device):
    print("\n===== (B) split-RoPE addcmul_ characterization =====")
    cases = [(1, 2048, 1, 64), (2, 257, 32, 64), (1, 129, 32, 128), (1, 512, 32, 128)]
    survivors = None
    per_variant_mismatch = {}
    sensitivity = {}
    for (B, S, nh, hd) in cases:
        for seed in range(6):
            x, cos, sin = make_rope_inputs(device, B, S, nh, hd, seed)
            y_ref = apply_split_rotary_emb_eager(x, (cos, sin))
            cands = rope_variants(x, cos, sin, nh, hd)
            ok_now = set()
            for name, y in cands.items():
                mm = n_mismatch(y, y_ref)
                per_variant_mismatch[name] = per_variant_mismatch.get(name, 0) + mm
                if mm == 0:
                    ok_now.add(name)
            survivors = ok_now if survivors is None else (survivors & ok_now)
            # sensitivity: do the variants actually differ from each other on this data?
            names = list(cands)
            diffs = sum(1 for a in range(len(names)) for b in range(a + 1, len(names))
                        if not biteq(cands[names[a]], cands[names[b]]))
            sensitivity[(B, S, nh, hd)] = diffs
    print("  Per-variant total mismatches vs eager (0 == bit-exact everywhere):")
    for name in sorted(per_variant_mismatch):
        print(f"    {name:24s} mismatches={per_variant_mismatch[name]}")
    print(f"  SURVIVORS (match eager on every case/seed): {sorted(survivors) if survivors else 'NONE'}")
    print(f"  variant-pair disagreement counts (sensitivity, must be >0 to be meaningful): {sensitivity}")
    return survivors, per_variant_mismatch, sensitivity


def main():
    print("===== Numerics characterization probe =====")
    print(f"python      : {platform.python_version()}")
    print(f"torch       : {torch.__version__}")
    print(f"cuda avail  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[FAIL] needs CUDA (run on the B200).")
        return 1
    print(f"cuda version: {torch.version.cuda}")
    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"capability  : {torch.cuda.get_device_capability(0)}")
    dev = torch.device("cuda")
    rms_surv, _, rstd_rep = characterize_rmsnorm(dev)
    rope_surv, _, rope_sens = characterize_rope(dev)
    print("\n===== VERDICT =====")
    print(f"RMSNorm bit-exact variant(s): {sorted(rms_surv) if rms_surv else 'NONE — fall back to ATen RMSNorm'}")
    print(f"RMSNorm rstd oracle         : {rstd_rep}")
    print(f"split-RoPE bit-exact variant: {sorted(rope_surv) if rope_surv else 'NONE'}")
    meaningful = all(v > 0 for v in rope_sens.values())
    print(f"split-RoPE sensitivity OK   : {meaningful} (variants disagree, so a match is informative)")
    ok = bool(rope_surv) and meaningful
    print(f"\nprobe status: {'OK' if ok else 'INCOMPLETE — inspect above'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
