# KDA Prompt: sgl_kernel_rmsnorm

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `sgl_kernel.rmsnorm`

Goal: optimize or replace this interface for the baidu/ERNIE-4.5-21B-A3B-PT serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `baidu/ERNIE-4.5-21B-A3B-PT`
- Model folder: `llm/ernie_45/b200`
- Category: `norm`
- Python interface: `sgl_kernel.rmsnorm`
- Captured call count: `56`
- Captured variants: `31`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `arg[0]: shape=[1, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[10124, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[12, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[14708, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[15, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[1543, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[15580, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[16, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[18, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[19, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[24, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[27, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`4`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(2560,)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1e-05", "scalars": ["arg[2]=1e-05"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [1, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [2560]}]}]`
   - kwargs: `{}`
2. label=`random_low`, calls=`1`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(38, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(2560,)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1e-05", "scalars": ["arg[2]=1e-05"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [38, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [2560]}]}]`
   - kwargs: `{}`
3. label=`random_mid`, calls=`2`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(27, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(2560,)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1e-05", "scalars": ["arg[2]=1e-05"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [27, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [2560]}]}]`
   - kwargs: `{}`
4. label=`random_mid`, calls=`1`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(12, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(2560,)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1e-05", "scalars": ["arg[2]=1e-05"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [12, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [2560]}]}]`
   - kwargs: `{}`
5. label=`random_mid`, calls=`1`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(15, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(2560,)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1e-05", "scalars": ["arg[2]=1e-05"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [15, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [2560]}]}]`
   - kwargs: `{}`
6. label=`random_mid`, calls=`1`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1543, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(2560,)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=1e-05", "scalars": ["arg[2]=1e-05"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[0]", "requires_grad": false, "shape": [1543, 2560]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [2560]}]}]`
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
