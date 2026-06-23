"""Build + load helper for the standalone build_tree extension, exposing an
ABI-agnostic call surface (`baseline`, `candidate`, `noop`) so the rest of the
harness (adapter.py, correctness.py, floor_probe.py) does not depend on the
underlying binding mechanism.

ABI: local direct-symbol TVM-FFI (`TVM_FFI_DLL_EXPORT_TYPED_FUNC` /
`tvm::ffi::TensorView`, destination-passing, current CUDA stream), matching the
`diffusion/kernels/*` pattern in this repo. The recovered baseline
(baseline/build_tree_baseline.cu) and the native-CUDA candidate
(solution/build_tree_candidate.cu) are compiled TOGETHER in ONE module with
SYMMETRIC flags (`-std=c++17 -O3` + native gencode, NO --use_fast_math), so both
sides share an identical registration/export/build style and call path. The three
wrapper functions below are the only entry points the harness uses; torch tensors
+ python ints are marshalled to TensorView / int64_t by tvm-ffi automatically.
"""

from __future__ import annotations

import functools
import pathlib

import torch

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent  # the kernel task folder
_BASELINE_CU = _ROOT / "baseline" / "build_tree_baseline.cu"
_CANDIDATE_CU = _ROOT / "solution" / "build_tree_candidate.cu"
_INCLUDE = _HERE / "csrc"


def _gencode_flag() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"


def candidate_compile_flags() -> dict:
    """Exact flag sets used to build BOTH sides (provenance record). Symmetric;
    no one-sided fast-math."""
    return {
        "extra_cflags": ["-std=c++17", "-O3"],
        "extra_cuda_cflags": ["-std=c++17", "-O3", _gencode_flag()],
        "ldlibs": ["-lc10", "-lc10_cuda", "-ltorch_cpu", "-ltorch_cuda"],
        "use_fast_math": False,
    }


@functools.lru_cache(maxsize=None)
def get_ext():
    """Build (once) and return the loaded TVM-FFI module exposing
    build_tree_baseline / build_tree_candidate / build_tree_noop."""
    from torch.utils import cpp_extension as torch_cpp
    from tvm_ffi.cpp import load

    include_paths = list(torch_cpp.include_paths(device_type="cuda"))
    library_paths = list(torch_cpp.library_paths(device_type="cuda"))
    flags = candidate_compile_flags()
    ldflags = [f"-L{p}" for p in library_paths]
    ldflags += [f"-Wl,-rpath,{p}" for p in library_paths]
    ldflags += flags["ldlibs"]

    return load(
        "build_tree_ext",
        cuda_files=[str(_BASELINE_CU), str(_CANDIDATE_CU)],
        extra_cflags=flags["extra_cflags"],
        extra_cuda_cflags=flags["extra_cuda_cflags"],
        extra_ldflags=ldflags,
        extra_include_paths=include_paths + [str(_INCLUDE)],
        build_directory=str(_HERE / ".build"),
    )


# ---- ABI-agnostic call surface (identical signatures for both sides) ----
# args: parent_list, selected_index, verified_seq_len, tree_mask, positions,
#       retrive_index, retrive_next_token, retrive_next_sibling,
#       topk, depth, draft_token_num, tree_mask_mode  -> None (in place)
def baseline(*args) -> None:
    get_ext().build_tree_baseline(*args)


def candidate(*args) -> None:
    get_ext().build_tree_candidate(*args)


def noop(verified_seq_len, draft_token_num) -> None:
    get_ext().build_tree_noop(verified_seq_len, draft_token_num)
