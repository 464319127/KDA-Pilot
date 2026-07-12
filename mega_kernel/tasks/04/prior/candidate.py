"""Native-CUDA candidate loader + host-side dispatch guards (strict-native policy).

Builds the workspace-owned CUDA kernels via CUDA extensions (nvcc, sm_100):
  - solution/mla_sparse_decode.cu  — two-stage split-KV sparse-MLA decode core
    (ported with attribution from the completed sibling run of this task slug,
    worktree glm_52__sglang_unified_attention_with_output-20260707-203446-78920)
  - solution/mla_wrapper_prep.cu   — fused DSA-wrapper prep (strided gather +
    interleaved RoPE + fp8 quant + KV store), new in this run

`native_path` is the cheap host-side guard (metadata/scalar checks only, no sync)
that decides whether a row is claimed by a native path; everything else falls back
WHOLESALE to the installed flashinfer baseline. A claimed row's execution path is
entirely workspace CUDA (no flashinfer / prebuilt ops).

Env toggles (candidate configuration only; never touches timing policy/workloads):
  MLA_TILE=<n>            stage-1 chunk size sweep (0 -> kernel default 32)
  MLA_NATIVE_WRAPPER=0    disable the wrapper native path (staged measurement)
"""
import functools
import os
import pathlib

import torch

_SOL = pathlib.Path(__file__).resolve().parent
_TILE = int(os.environ.get("MLA_TILE", "0"))  # 0 -> kernel default (32); sweep via env
_DEFAULT_TILE = 32
_WRAPPER_ENABLED = os.environ.get("MLA_NATIVE_WRAPPER", "1") != "0"
# Largest batch/token-count the native paths claim. The kernels are generic over B
# (stage-1 grid (S,H,B); prep grid (B)); the default reflects the measured dispatch
# table in docs/dispatch.md. MLA_NATIVE_MAX_B overrides for probe/ablation runs.
_MAX_B = int(os.environ.get("MLA_NATIVE_MAX_B", "1"))


@functools.lru_cache(maxsize=1)
def _ext_decode():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
    from torch.utils.cpp_extension import load
    return load(
        name="mla_sparse_decode_ext",
        sources=[str(_SOL / "mla_sparse_decode.cu")],
        extra_cuda_cflags=["-O3", "-lineinfo"],  # no fast-math (symmetric with baseline)
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def _ext_prep():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
    from torch.utils.cpp_extension import load
    return load(
        name="mla_wrapper_prep_ext",
        sources=[str(_SOL / "mla_wrapper_prep.cu")],
        extra_cuda_cflags=["-O3", "-lineinfo"],  # no fast-math (symmetric with baseline)
        verbose=False,
    )


def _scratch_bytes(max_seq_len: int, batch: int) -> int:
    # fp32 partials [part_m|part_l|part_o] = B*H*S*(512+2) floats.
    tile = _TILE if _TILE > 0 else _DEFAULT_TILE
    lmax = min(int(max_seq_len), 2048)
    s_chunks = (lmax + tile - 1) // tile
    return batch * 8 * s_chunks * (512 + 2) * 4


def _make_scratch_tensor(device, max_seq_len: int, batch: int) -> torch.Tensor:
    """Allocate the native path's PRIVATE fp32-partials scratch at case-setup time
    (untimed). The native kernels must NOT reuse the row's captured workspace_buffer:
    the trtllm-gen cubins keep live multi-CTA-split state in that buffer between
    launches, and overwriting it makes the NEXT baseline cubin launch illegal-access
    (reproduced at B=16, L=2048: baseline OK -> native OK -> baseline crash; isolation
    matrix in docs/run_log.md). A private scratch removes the coupling entirely.
    seq_lens VALUES and slot ids are intentionally not guarded host-side (that would
    cost a device sync): the kernel itself clamps L = min(seq_lens, top_k) and masks
    slot < 0 / >= num_slots, validated by the short-seq and boundary regression rows
    plus the independent oracle's masking cases."""
    return torch.empty(_scratch_bytes(max_seq_len, batch), dtype=torch.uint8, device=device)


def _native_decode_eligible(kwargs: dict, out: torch.Tensor) -> bool:
    """Cheap host-side guard for the direct-decode native fast path (B=1). Enforces the full
    documented dispatch contract (docs/dispatch.md): device SM100, exact shapes/dtypes/contiguity
    of query/kv_cache/block_tables/seq_lens/out, scalar metadata, and a uint8 workspace on the same
    device large enough for the fp32 partials. Any mismatch returns False -> baseline fallback."""
    q = kwargs.get("query"); kv = kwargs.get("kv_cache")
    bt = kwargs.get("block_tables"); sl = kwargs.get("seq_lens")
    if not all(torch.is_tensor(t) for t in (q, kv, bt, sl)):
        return False
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] < 10:  # SM100+ (B200)
        return False
    b = int(q.shape[0])
    if not (1 <= b <= _MAX_B):                           # measured native buckets only
        return False
    # Short-sequence B>1 measured 0.46x (reg_decode_b16_shortseq_65: too few chunks to
    # fill the GPU while the stage-2 combine cost stays fixed) -> fallback. B=1 wins at
    # every captured length (the baseline cubin is launch-bound there regardless).
    if b > 1 and min(int(kwargs.get("max_seq_len", 0)), 2048) < 512:
        return False
    if q.dtype != torch.float8_e4m3fn or kv.dtype != torch.float8_e4m3fn:
        return False
    if tuple(q.shape) != (b, 1, 8, 576):
        return False
    if kv.dim() != 4 or kv.shape[1] != 1 or kv.shape[2] != 64 or kv.shape[3] != 576:
        return False
    if bt.dtype != torch.int32 or bt.dim() != 3 or bt.shape[0] != b or bt.shape[1] != 1 or bt.shape[-1] != 2048:
        return False
    if sl.dtype != torch.int32 or tuple(sl.shape) != (b,):    # seq_lens int32 [B]
        return False
    if not (q.is_contiguous() and kv.is_contiguous() and bt.is_contiguous()
            and sl.is_contiguous() and out.is_contiguous()):
        return False
    if not (kv.device == q.device and bt.device == q.device      # single-device dispatch
            and sl.device == q.device and out.device == q.device):
        return False
    if int(kwargs.get("kv_lora_rank", 0)) != 512 or int(kwargs.get("qk_rope_head_dim", 0)) != 64:
        return False
    if int(kwargs.get("qk_nope_head_dim", 0)) != 192:
        return False
    if kwargs.get("backend", "trtllm-gen") != "trtllm-gen":
        return False
    if int(kwargs.get("sparse_mla_top_k", 0)) != 2048:
        return False
    if kwargs.get("skip_softmax_threshold_scale_factor") is not None:
        return False
    if tuple(out.shape) != (b, 1, 8, 512) or out.dtype != torch.bfloat16:
        return False
    return True


def _native_wrapper_eligible(b: dict, out: torch.Tensor) -> bool:
    """Guard for the fused native T=1 DSA-wrapper path: fused prep kernel + native decode
    core, entirely workspace CUDA. Enforces the captured wrapper contract (dims 512/64,
    top_k 2048, is_neox False, fp8 pool with page 64) and every layout the two kernels rely
    on. Any mismatch -> wholesale baseline fallback."""
    if not _WRAPPER_ENABLED:
        return False
    q_nope = b.get("q_nope"); q_rope = b.get("q_rope")
    k_nope = b.get("k_nope"); k_rope = b.get("k_rope")
    pool = b.get("kv_pool_candidate"); bt = b.get("block_tables"); sl = b.get("seq_lens")
    qn = b.get("query_native"); cs = b.get("cos_sin_cache"); pos = b.get("pos_ids")
    slots = b.get("current_slots")
    if not all(torch.is_tensor(t) for t in (q_nope, q_rope, k_nope, k_rope, pool, bt, sl, qn, cs, pos, slots)):
        return False
    if not q_nope.is_cuda or torch.cuda.get_device_capability(q_nope.device)[0] < 10:
        return False
    n = int(b.get("n", 0))
    if not (1 <= n <= _MAX_B):                                # measured native buckets only
        return False
    if int(b.get("heads", 0)) != 8 or int(b.get("kv_lora", 0)) != 512:
        return False
    if int(b.get("rope_dim", 0)) != 64 or int(b.get("top_k", 0)) != 2048 or int(b.get("dim", 0)) != 576:
        return False
    if bool(b.get("is_neox", True)):                          # captured contract: is_neox=False
        return False
    if q_nope.dtype != torch.bfloat16 or q_rope.dtype != torch.bfloat16:
        return False
    if k_nope.dtype != torch.bfloat16 or k_rope.dtype != torch.bfloat16:
        return False
    if q_nope.stride(2) != 1 or q_rope.stride(2) != 1 or k_nope.stride(1) != 1 or k_rope.stride(1) != 1:
        return False                                          # kernels require unit innermost stride
    if pool.dtype != torch.float8_e4m3fn or not pool.is_contiguous():
        return False
    if pool.dim() != 4 or pool.shape[1] != 1 or pool.shape[2] != 64 or pool.shape[3] != 576:
        return False
    if bt.dtype != torch.int32 or not bt.is_contiguous() or tuple(bt.shape) != (n, 1, 2048):
        return False
    if sl.dtype != torch.int32 or tuple(sl.shape) != (n,):
        return False
    if qn.dtype != torch.float8_e4m3fn or not qn.is_contiguous() or tuple(qn.shape) != (n, 1, 8, 576):
        return False
    if cs.dtype != torch.float32 or not cs.is_contiguous() or cs.shape[-1] != 64:
        return False
    # fused_wrapper_prep indexes pos_ids[t] / current_slots[t] as flat contiguous
    # (n,) arrays — a strided view or extra dims would silently read wrong RoPE
    # positions / write KV rows to wrong slots, so reject them here (fallback).
    if pos.dtype != torch.int32 or tuple(pos.shape) != (n,) or not pos.is_contiguous():
        return False
    if slots.dtype != torch.int64 or tuple(slots.shape) != (n,) or not slots.is_contiguous():
        return False
    devices = {t.device for t in (q_nope, q_rope, k_nope, k_rope, pool, bt, sl, qn, cs, pos, slots, out)}
    if len(devices) != 1:
        return False
    if tuple(out.shape) != (n, 1, 8, 512) or out.dtype != torch.bfloat16 or not out.is_contiguous():
        return False
    return True


def native_path(call: str, payload: dict, out: torch.Tensor) -> str:
    """Resolve the native route for one case: '' (fallback), 'decode', or 'wrapper'."""
    if call == "decode":
        return "decode" if _native_decode_eligible(payload, out) else ""
    if call == "dsa_wrapper":
        return "wrapper" if _native_wrapper_eligible(payload, out) else ""
    return ""


def make_scratch(route: str, payload: dict) -> torch.Tensor:
    """Allocate the private native scratch for an eligible route (untimed setup;
    stored by the adapter OUTSIDE the payload so the baseline's kwargs splat never
    sees it)."""
    if route == "decode":
        q = payload["query"]
        return _make_scratch_tensor(q.device, int(payload.get("max_seq_len", 0)), int(q.shape[0]))
    if route == "wrapper":
        return _make_scratch_tensor(payload["q_nope"].device,
                                    int(payload["max_seq_len_host"]), int(payload["n"]))
    raise ValueError(f"no scratch for route {route!r}")


def run_native_decode(kwargs: dict, out: torch.Tensor, scratch: torch.Tensor) -> None:
    _ext_decode().mla_sparse_decode(
        kwargs["query"], kwargs["kv_cache"], kwargs["block_tables"],
        kwargs["seq_lens"], scratch, out,
        float(kwargs["bmm1_scale"]), int(kwargs["max_seq_len"]), _TILE,
    )


def run_native_wrapper(b: dict, out: torch.Tensor, scratch: torch.Tensor) -> None:
    pool = b["kv_pool_candidate"]
    _ext_prep().mla_wrapper_prep(
        b["q_nope"], b["q_rope"], b["k_nope"], b["k_rope"],
        b["cos_sin_cache"], b["pos_ids"], b["current_slots"],
        b["query_native"], pool.view(-1, b["dim"]),
    )
    _ext_decode().mla_sparse_decode(
        b["query_native"], pool, b["block_tables"], b["seq_lens"],
        scratch, out,
        0.0625, int(b["max_seq_len_host"]), _TILE,
    )
