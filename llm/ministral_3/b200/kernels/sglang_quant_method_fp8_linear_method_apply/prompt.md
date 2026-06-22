# KDA Prompt: sglang_quant_method_fp8_linear_method_apply

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `sglang.quant_method.Fp8LinearMethod.apply`

Goal: optimize or replace this interface for the mistralai/Ministral-3-14B-Instruct-2512 serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `mistralai/Ministral-3-14B-Instruct-2512`
- Model folder: `llm/ministral_3/b200`
- Category: `quantization`
- Python interface: `sglang.quant_method.Fp8LinearMethod.apply`
- Captured call count: `9148`
- Captured variants: `124`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `arg[1]: shape=[1, 16384], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[1, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[1, 5120], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[12, 16384], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[12, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[12, 5120], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[14685, 16384], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[14685, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[14685, 5120], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[15, 16384], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[15, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[15, 5120], dtype=bfloat16, device=cuda:0, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`207`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=MergedColumnParallelLinear(\n      repr=MergedColumnParallelLinear(in_features=5120, output_features=32768, bias=False, tp_size=1, gather_output=False)\n    )\n  arg[1]=Tensor(\n      shape=(1, 5120)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=None", "scalars": ["arg[0]=MergedColumnParallelLinear(", "repr=MergedColumnParallelLinear(in_features=5120, output_features=32768, bias=False, tp_size=1, gather_output=False)", "arg[2]=None"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 5120]}]}]`
   - kwargs: `{}`
2. label=`random_low`, calls=`207`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=QKVParallelLinear(\n      repr=QKVParallelLinear(in_features=5120, output_features=6144, bias=False, tp_size=1, gather_output=False)\n    )\n  arg[1]=Tensor(\n      shape=(1, 5120)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=None", "scalars": ["arg[0]=QKVParallelLinear(", "repr=QKVParallelLinear(in_features=5120, output_features=6144, bias=False, tp_size=1, gather_output=False)", "arg[2]=None"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 5120]}]}]`
   - kwargs: `{}`
3. label=`random_low`, calls=`207`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=16384, output_features=5120, bias=False, tp_size=1, reduce_results=True)\n    )\n  arg[1]=Tensor(\n      shape=(1, 16384)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=16384, output_features=5120, bias=False, tp_size=1, reduce_results=True)", "bias=None"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 16384]}]}]`
   - kwargs: `{}`
4. label=`random_low`, calls=`207`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=4096, output_features=5120, bias=False, tp_size=1, reduce_results=True)\n    )\n  arg[1]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=4096, output_features=5120, bias=False, tp_size=1, reduce_results=True)", "bias=None"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 4096]}]}]`
   - kwargs: `{}`
5. label=`random_low`, calls=`40`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=MergedColumnParallelLinear(\n      repr=MergedColumnParallelLinear(in_features=5120, output_features=32768, bias=False, tp_size=1, gather_output=False)\n    )\n  arg[1]=Tensor(\n      shape=(38, 5120)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=None", "scalars": ["arg[0]=MergedColumnParallelLinear(", "repr=MergedColumnParallelLinear(in_features=5120, output_features=32768, bias=False, tp_size=1, gather_output=False)", "arg[2]=None"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [38, 5120]}]}]`
   - kwargs: `{}`
6. label=`random_low`, calls=`40`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=QKVParallelLinear(\n      repr=QKVParallelLinear(in_features=5120, output_features=6144, bias=False, tp_size=1, gather_output=False)\n    )\n  arg[1]=Tensor(\n      shape=(38, 5120)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=None", "scalars": ["arg[0]=QKVParallelLinear(", "repr=QKVParallelLinear(in_features=5120, output_features=6144, bias=False, tp_size=1, gather_output=False)", "arg[2]=None"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [38, 5120]}]}]`
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
