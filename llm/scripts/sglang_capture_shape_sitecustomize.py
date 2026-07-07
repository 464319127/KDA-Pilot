"""Record selected SGLang kernel API call contracts as JSONL.

Use as a temporary ``sitecustomize.py`` by placing this file in a directory on
``PYTHONPATH`` before launching SGLang. It records tensor metadata only: no
tensor values are copied or saved.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any


OUT_TEMPLATE = os.environ.get("KDA_CAPTURE_SHAPES_JSONL")
MAX_CALLS_PER_FUNC = int(os.environ.get("KDA_CAPTURE_MAX_CALLS_PER_FUNC", "256"))

TARGETS: dict[str, tuple[str, ...]] = {
    "sglang.srt.layers.quantization.fp8_kernel": (
        "deep_gemm_fp8_fp8_bf16_nt",
        "sglang_per_token_group_quant_fp8",
        "sglang_per_token_group_quant_fp8_row_padded",
        "per_token_group_quant_fp8",
        "sglang_per_token_group_quant_8bit",
        "per_token_group_quant_8bit",
    ),
    "sglang.srt.layers.radix_attention": (
        "unified_attention_with_output",
        "breakable_unified_attention_with_output",
    ),
    "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla": (
        "bmm_fp8",
        "_bmm_fp8_op",
        "mla_bmm_then_unified_attention",
        "bcg_mla_bmm_then_unified_attention",
    ),
    "sglang.srt.layers.quantization.fp8_utils": (
        "flashinfer_bmm_fp8",
    ),
    "sglang.srt.layers.attention.deepseek_v4_backend": (
        "DeepseekV4AttnBackend.forward",
        "DeepseekV4AttnBackend._forward_prefill_sparse",
    ),
    "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix": (
        "DeepseekV4HipRadixBackend.forward",
    ),
    "sglang.srt.layers.attention.dsv4.indexer": (
        "fp8_paged_mqa_logits_torch",
        "fp8_paged_mqa_logits_torch_sm120",
        "_aiter_fp8_paged_mqa_logits",
    ),
    "sglang.srt.layers.attention.dsa_backend": (
        "DeepseekSparseAttnBackend.forward_extend",
        "DeepseekSparseAttnBackend.forward_decode",
        "DeepseekSparseAttnBackend._forward_fa3",
        "DeepseekSparseAttnBackend._forward_flashmla_sparse",
        "DeepseekSparseAttnBackend._forward_flashmla_kv",
        "DeepseekSparseAttnBackend._forward_standard_mha",
        "DeepseekSparseAttnBackend._forward_trtllm",
    ),
    "sgl_kernel.flash_mla": (
        "flash_mla_with_kvcache",
        "flash_mla_sparse_fwd",
    ),
    "sglang.srt.layers.attention.flash_mla_sm120": (
        "flash_mla_with_kvcache_sm120",
    ),
    "flashinfer.prefill": (
        "trtllm_ragged_attention_deepseek",
    ),
    "flashinfer.decode": (
        "trtllm_batch_decode_with_kv_cache_mla",
    ),
}

_lock = threading.Lock()
_counts: dict[str, int] = {}
_wrapped: set[str] = set()


def _out_path() -> Path | None:
    if not OUT_TEMPLATE:
        return None
    return Path(OUT_TEMPLATE.replace("%p", str(os.getpid())))


def _rank_env() -> dict[str, str | None]:
    keys = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "TP_RANK",
        "CUDA_VISIBLE_DEVICES",
        "SGLANG_CAPTURE_SCENARIO",
    )
    return {key: os.environ.get(key) for key in keys}


def _tensor_meta(value: Any) -> dict[str, Any] | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.is_tensor(value):
        return None
    device = value.device
    return {
        "kind": "tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(device),
        "device_type": device.type,
        "device_index": device.index,
        "stride": list(value.stride()),
        "storage_offset": int(value.storage_offset()),
        "is_contiguous": bool(value.is_contiguous()),
        "requires_grad": bool(value.requires_grad),
        "numel": int(value.numel()),
    }


def _summarize(value: Any, depth: int = 0) -> Any:
    tensor = _tensor_meta(value)
    if tensor is not None:
        return tensor
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= 3:
        return {"kind": type(value).__name__, "repr": repr(value)[:160]}
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [_summarize(item, depth + 1) for item in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {
                str(key): _summarize(item, depth + 1)
                for key, item in list(value.items())[:64]
            },
        }
    return {"kind": type(value).__name__, "repr": repr(value)[:160]}


def _append_record(record: dict[str, Any]) -> None:
    path = _out_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _is_torch_compiling() -> bool:
    try:
        import torch

        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling) and is_compiling():
            return True
    except Exception:
        pass
    try:
        import torch._dynamo

        return bool(torch._dynamo.is_compiling())
    except Exception:
        return False


def _resolve_target(module: Any, target_name: str) -> tuple[Any, str] | None:
    parts = target_name.split(".")
    parent = module
    for part in parts[:-1]:
        if not hasattr(parent, part):
            return None
        parent = getattr(parent, part)
    return parent, parts[-1]


def _wrap_function(module: Any, module_name: str, target_name: str) -> None:
    resolved = _resolve_target(module, target_name)
    if resolved is None:
        return
    parent, attr_name = resolved
    if not hasattr(parent, attr_name):
        return
    func = getattr(parent, attr_name)
    if not callable(func) or getattr(func, "_kda_shape_capture_wrapped", False):
        return
    full_name = f"{module_name}.{target_name}"
    if full_name in _wrapped:
        return

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _is_torch_compiling():
            return func(*args, **kwargs)
        count = _counts.get(full_name, 0)
        should_record = count < MAX_CALLS_PER_FUNC
        if should_record:
            _counts[full_name] = count + 1
            record: dict[str, Any] = {
                "event": "call",
                "time": time.time(),
                "pid": os.getpid(),
                "call_index": count + 1,
                "function": full_name,
                "rank_env": _rank_env(),
                "args": [_summarize(arg) for arg in args],
                "kwargs": {key: _summarize(val) for key, val in kwargs.items()},
            }
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                record["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                _append_record(record)
                raise
            record["result"] = _summarize(result)
            _append_record(record)
            return result
        return func(*args, **kwargs)

    setattr(wrapper, "_kda_shape_capture_wrapped", True)
    setattr(parent, attr_name, wrapper)
    _wrapped.add(full_name)


def _wrap_module(module: Any, module_name: str) -> None:
    for attr_name in TARGETS.get(module_name, ()):
        _wrap_function(module, module_name, attr_name)


class _CaptureLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, module_name: str):
        self._wrapped = wrapped
        self._module_name = module_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Any:
        create = getattr(self._wrapped, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module: Any) -> None:
        self._wrapped.exec_module(module)
        _wrap_module(module, self._module_name)


class _CaptureFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in TARGETS:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            spec.loader = _CaptureLoader(spec.loader, fullname)
            return spec
        return None


if OUT_TEMPLATE:
    sys.meta_path.insert(0, _CaptureFinder())
    for _module_name, _module in list(sys.modules.items()):
        if _module_name in TARGETS:
            _wrap_module(_module, _module_name)

    if os.environ.get("KDA_CAPTURE_TORCH_BMM") == "1":
        try:
            import torch

            if not getattr(torch.bmm, "_kda_shape_capture_wrapped", False):
                _torch_bmm = torch.bmm

                @functools.wraps(_torch_bmm)
                def _bmm_wrapper(*args: Any, **kwargs: Any) -> Any:
                    if _is_torch_compiling():
                        return _torch_bmm(*args, **kwargs)
                    full_name = "torch.bmm"
                    count = _counts.get(full_name, 0)
                    should_record = count < MAX_CALLS_PER_FUNC
                    if should_record:
                        _counts[full_name] = count + 1
                        record: dict[str, Any] = {
                            "event": "call",
                            "time": time.time(),
                            "pid": os.getpid(),
                            "call_index": count + 1,
                            "function": full_name,
                            "rank_env": _rank_env(),
                            "args": [_summarize(arg) for arg in args],
                            "kwargs": {
                                key: _summarize(val) for key, val in kwargs.items()
                            },
                        }
                        try:
                            result = _torch_bmm(*args, **kwargs)
                        except Exception as exc:
                            record["exception"] = {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            }
                            _append_record(record)
                            raise
                        record["result"] = _summarize(result)
                        _append_record(record)
                        return result
                    return _torch_bmm(*args, **kwargs)

                setattr(_bmm_wrapper, "_kda_shape_capture_wrapped", True)
                torch.bmm = _bmm_wrapper
        except Exception:
            pass
