# KDA Prompt: sgl_kernel_moe_sum_reduce

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `sgl_kernel.moe_sum_reduce`

Goal: optimize or replace this interface for the baidu/ERNIE-4.5-21B-A3B-PT serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `baidu/ERNIE-4.5-21B-A3B-PT`
- Model folder: `llm/ernie_45/b200`
- Category: `moe`
- Python interface: `sgl_kernel.moe_sum_reduce`
- Captured call count: `621`
- Captured variants: `16`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `arg[0]: shape=[10124, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[14708, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[1543, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[15580, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[34, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[38, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[44, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[53, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[61, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[6294, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[70, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[72, 6, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(38, 6, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(38, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1.0", "scalars": ["arg[2]=1.0"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [38, 6, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [38, 2560]}]}]`
   - kwargs: `{}`
2. label=`random_mid`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1543, 6, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1543, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1.0", "scalars": ["arg[2]=1.0"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [1543, 6, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1543, 2560]}]}]`
   - kwargs: `{}`
3. label=`random_mid`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(15580, 6, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(15580, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1.0", "scalars": ["arg[2]=1.0"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [15580, 6, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [15580, 2560]}]}]`
   - kwargs: `{}`
4. label=`random_mid`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(38, 6, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(38, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1.0", "scalars": ["arg[2]=1.0"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [38, 6, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [38, 2560]}]}]`
   - kwargs: `{}`
5. label=`random_high`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(14708, 6, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(14708, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1.0", "scalars": ["arg[2]=1.0"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [14708, 6, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [14708, 2560]}]}]`
   - kwargs: `{}`
6. label=`random_high`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(34, 6, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(34, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1.0", "scalars": ["arg[2]=1.0"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [34, 6, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [34, 2560]}]}]`
   - kwargs: `{}`

Full structured args/kwargs/result records are in `docs/evidence.json`.

## Required First Milestone

1. Copy the upstream SGLang source files needed for this exact interface into `baseline/`.
2. Record upstream URL, commit, and copied files in `docs/baseline_source.md`.
3. Expose the copied baseline through a local low-overhead ABI.
4. Expose the candidate through the exact same ABI in `solution/`.
5. Build correctness tests for every retained captured variant or an explicitly justified representative subset.
6. Benchmark baseline and candidate on an idle B200 with the same shapes, dtypes, devices, contiguity, and scalar parameters.
- Unsupported shapes or parameter combinations must fall back to the recovered SGLang baseline.

Do not import, patch, or monkey-patch a live SGLang server during correctness or benchmark runs.
