# KDA Prompt: jit_kernel_activation_run_activation_inplace

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `jit_kernel.activation._run_activation_inplace`

Goal: optimize or replace this interface for the google/gemma-4-26B-A4B-it serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `google/gemma-4-26B-A4B-it`
- Model folder: `llm/gemma4/b200`
- Category: `other`
- Python interface: `jit_kernel.activation._run_activation_inplace`
- Captured call count: `3300`
- Captured variants: `62`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `arg[1]: shape=[1, 4224], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[101384, 1408], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[103, 4224], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[11, 4224], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[12, 4224], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[120, 1408], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[12673, 4224], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[137632, 1408], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[15, 4224], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[168, 1408], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[17204, 4224], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[176, 1408], dtype=bfloat16, device=cuda:0, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`120`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='gelu'\n  arg[1]=Tensor(\n      shape=(8, 1408)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(8, 704)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='gelu'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [8, 1408]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [8, 704]}]}]`
   - kwargs: `{}`
2. label=`random_low`, calls=`120`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='gelu_tanh'\n  arg[1]=Tensor(\n      shape=(1, 4224)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 2112)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='gelu_tanh'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 4224]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [1, 2112]}]}]`
   - kwargs: `{}`
3. label=`random_low`, calls=`30`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='gelu'\n  arg[1]=Tensor(\n      shape=(824, 1408)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(824, 704)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='gelu'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [824, 1408]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [824, 704]}]}]`
   - kwargs: `{}`
4. label=`random_low`, calls=`30`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='gelu_tanh'\n  arg[1]=Tensor(\n      shape=(103, 4224)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(103, 2112)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='gelu_tanh'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [103, 4224]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [103, 2112]}]}]`
   - kwargs: `{}`
5. label=`random_mid`, calls=`30`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='gelu'\n  arg[1]=Tensor(\n      shape=(101384, 1408)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(101384, 704)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='gelu'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [101384, 1408]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [101384, 704]}]}]`
   - kwargs: `{}`
6. label=`random_mid`, calls=`30`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]='gelu'\n  arg[1]=Tensor(\n      shape=(120, 1408)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(120, 704)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )", "scalars": ["arg[0]='gelu'"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [120, 1408]}, {"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[2]", "requires_grad": false, "shape": [120, 704]}]}]`
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
