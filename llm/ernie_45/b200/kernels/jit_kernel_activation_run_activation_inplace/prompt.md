# KDA Prompt: jit_kernel_activation_run_activation_inplace

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `jit_kernel.activation._run_activation_inplace`

Goal: optimize or replace this interface for the baidu/ERNIE-4.5-21B-A3B-PT serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `baidu/ERNIE-4.5-21B-A3B-PT`
- Model folder: `llm/ernie_45/b200`
- Category: `other`
- Python interface: `jit_kernel.activation._run_activation_inplace`
- Captured call count: `3116`
- Captured variants: `93`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `arg[1]: shape=[1, 24576], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[1, 6144], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[10124, 24576], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[10124, 6144], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[108, 3072], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[114, 3072], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[12, 24576], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[12, 6144], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[144, 3072], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[14708, 24576], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[14708, 6144], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[15, 24576], dtype=bfloat16, device=cuda:0, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`126`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='silu'\n  arg[1]=Tensor(\n      shape=(1, 6144)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 3072)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='silu'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 6144]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [1, 3072]}]}]`
   - kwargs: `{}`
2. label=`random_low`, calls=`126`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='silu'\n  arg[1]=Tensor(\n      shape=(6, 3072)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(6, 1536)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='silu'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [6, 3072]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [6, 1536]}]}]`
   - kwargs: `{}`
3. label=`random_low`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='silu'\n  arg[1]=Tensor(\n      shape=(228, 3072)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(228, 1536)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='silu'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [228, 3072]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [228, 1536]}]}]`
   - kwargs: `{}`
4. label=`random_low`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='silu'\n  arg[1]=Tensor(\n      shape=(38, 6144)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(38, 3072)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='silu'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [38, 6144]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [38, 3072]}]}]`
   - kwargs: `{}`
5. label=`random_low`, calls=`4`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='silu'\n  arg[1]=Tensor(\n      shape=(1, 24576)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 12288)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='silu'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 24576]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [1, 12288]}]}]`
   - kwargs: `{}`
6. label=`random_low`, calls=`1`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='silu'\n  arg[1]=Tensor(\n      shape=(38, 24576)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(38, 12288)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='silu'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [38, 24576]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [38, 12288]}]}]`
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
