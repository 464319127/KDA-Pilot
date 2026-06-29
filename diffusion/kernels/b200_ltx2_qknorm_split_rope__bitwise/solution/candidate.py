"""Candidate entry point for the LTX2 qknorm + split-RoPE task.

run_candidate(inputs, outputs) writes q_out -> outputs[0], k_out -> outputs[1] in
place. Validation runs on EVERY call (no caching) and raises ValueError BEFORE any
RMSNorm/kernel launch on any unsupported configuration, leaving outputs untouched.

Staged correctness floor (see docs/numerics_characterization.md): ATen RMSNorm
(bit-exact to the eager oracle by construction) + one bit-exact split-RoPE CUDA
kernel (A1 sequence) per side. A fully-fused custom RMSNorm is a later optimization.
"""

import torch

from solution.build import load_candidate_module

_BF16 = torch.bfloat16


def _reject(msg):
    raise ValueError(f"unsupported configuration: {msg}")


def _validate(inputs, outputs):
    if int(inputs.get("tp_world_size", 1)) != 1:
        _reject(f"tensor-parallel world size {inputs.get('tp_world_size')} != 1")

    q, k = inputs["q"], inputs["k"]
    eps = float(inputs["eps"])
    num_heads = int(inputs["num_heads"])
    head_dim = int(inputs["head_dim"])

    # Norm modules: EXACT torch.nn.RMSNorm (subclasses rejected), matching eps, bf16 weight.
    for name in ("q_norm", "k_norm"):
        m = inputs[name]
        if type(m) is not torch.nn.RMSNorm:
            _reject(f"{name} is {type(m).__name__}, not torch.nn.RMSNorm")
        if m.eps is None or float(m.eps) != eps:
            _reject(f"{name}.eps {m.eps} != {eps}")
        if m.weight.dtype != _BF16:
            _reject(f"{name}.weight dtype {m.weight.dtype} != torch.bfloat16")

    # q/k tensors.
    for name, x in (("q", q), ("k", k)):
        if x.dtype != _BF16:
            _reject(f"{name} dtype {x.dtype} != torch.bfloat16")
        if x.dim() != 3:
            _reject(f"{name} ndim {x.dim()} != 3")
        if not x.is_contiguous():
            _reject(f"{name} is not contiguous")
        if not x.is_cuda:
            _reject(f"{name} is not on CUDA")
    if q.device != k.device:
        _reject(f"q device {q.device} != k device {k.device}")
    if q.shape[-1] != num_heads * head_dim:
        _reject(f"q hidden {q.shape[-1]} != num_heads*head_dim ({num_heads}*{head_dim})")
    if k.shape[-1] != num_heads * head_dim:
        _reject(f"k hidden {k.shape[-1]} != num_heads*head_dim ({num_heads}*{head_dim})")

    r = head_dim // 2

    def _check_freq(name, t, x):
        if t.dtype != _BF16:
            _reject(f"{name} dtype {t.dtype} != torch.bfloat16")
        if t.dim() != 4:
            _reject(f"{name} ndim {t.dim()} != 4 (split RoPE layout)")
        if t.stride(-1) != 1:
            _reject(f"{name} last-dim stride {t.stride(-1)} != 1")
        if t.device != x.device:
            _reject(f"{name} device {t.device} != {x.device}")
        want = (x.shape[0], num_heads, x.shape[1], r)
        if tuple(t.shape) != want:
            _reject(f"{name} shape {tuple(t.shape)} != {want}")

    _check_freq("q_cos", inputs["q_cos"], q)
    _check_freq("q_sin", inputs["q_sin"], q)
    _check_freq("k_cos", inputs["k_cos"], k)
    _check_freq("k_sin", inputs["k_sin"], k)

    # Output buffers: 2 preallocated bf16 contiguous tensors matching q/k.
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        _reject("outputs must be a list/tuple of exactly 2 tensors")
    for name, o, ref in (("out[0]", outputs[0], q), ("out[1]", outputs[1], k)):
        if not torch.is_tensor(o):
            _reject(f"{name} is not a tensor")
        if o.dtype != _BF16:
            _reject(f"{name} dtype {o.dtype} != torch.bfloat16")
        if tuple(o.shape) != tuple(ref.shape):
            _reject(f"{name} shape {tuple(o.shape)} != {tuple(ref.shape)}")
        if not o.is_contiguous():
            _reject(f"{name} is not contiguous")
        if o.device != ref.device:
            _reject(f"{name} device {o.device} != {ref.device}")


def run_candidate(inputs, outputs):
    _validate(inputs, outputs)
    mod = load_candidate_module()
    # ATen RMSNorm over full H: bit-exact to the eager oracle (same module call).
    q_normed = inputs["q_norm"](inputs["q"])
    k_normed = inputs["k_norm"](inputs["k"])
    mod.ltx2_split_rope_candidate(q_normed, inputs["q_cos"], inputs["q_sin"], outputs[0])
    mod.ltx2_split_rope_candidate(k_normed, inputs["k_cos"], inputs["k_sin"], outputs[1])
