# KDA Prompt: sglang_quant_method_unquantized_linear_method_apply

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `sglang.quant_method.UnquantizedLinearMethod.apply`

Goal: optimize or replace this interface for the tencent/Hy3-preview serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `tencent/Hy3-preview`
- Model folder: `llm/hunyuan3_preview/b200`
- Category: `quantization`
- Python interface: `sglang.quant_method.UnquantizedLinearMethod.apply`
- Captured call count: `177312`
- Captured variants: `1736`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `arg[1]: shape=[1, 1024], dtype=float16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[1, 1024], dtype=float16, device=cuda:1, contiguous=True`
- `arg[1]: shape=[1, 1024], dtype=float16, device=cuda:2, contiguous=True`
- `arg[1]: shape=[1, 1024], dtype=float16, device=cuda:3, contiguous=True`
- `arg[1]: shape=[1, 1024], dtype=float16, device=cuda:4, contiguous=True`
- `arg[1]: shape=[1, 1024], dtype=float16, device=cuda:5, contiguous=True`
- `arg[1]: shape=[1, 1024], dtype=float16, device=cuda:6, contiguous=True`
- `arg[1]: shape=[1, 1024], dtype=float16, device=cuda:7, contiguous=True`
- `arg[1]: shape=[1, 1664], dtype=float16, device=cuda:0, contiguous=True`
- `arg[1]: shape=[1, 1664], dtype=float16, device=cuda:1, contiguous=True`
- `arg[1]: shape=[1, 1664], dtype=float16, device=cuda:2, contiguous=True`
- `arg[1]: shape=[1, 1664], dtype=float16, device=cuda:3, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`364`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)\n    )\n  arg[1]=Tensor(\n      shape=(1, 1024)\n      dtype=torch.float16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)", "bias=None"], "tensors": [{"device": "cuda:0", "dtype": "float16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 1024]}]}]`
   - kwargs: `{}`
2. label=`random_low`, calls=`364`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)\n    )\n  arg[1]=Tensor(\n      shape=(1, 1024)\n      dtype=torch.float16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)", "bias=None"], "tensors": [{"device": "cuda:1", "dtype": "float16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 1024]}]}]`
   - kwargs: `{}`
3. label=`random_low`, calls=`364`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)\n    )\n  arg[1]=Tensor(\n      shape=(1, 1024)\n      dtype=torch.float16\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)", "bias=None"], "tensors": [{"device": "cuda:2", "dtype": "float16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 1024]}]}]`
   - kwargs: `{}`
4. label=`random_low`, calls=`364`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)\n    )\n  arg[1]=Tensor(\n      shape=(1, 1024)\n      dtype=torch.float16\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)", "bias=None"], "tensors": [{"device": "cuda:3", "dtype": "float16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 1024]}]}]`
   - kwargs: `{}`
5. label=`random_low`, calls=`364`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)\n    )\n  arg[1]=Tensor(\n      shape=(1, 1024)\n      dtype=torch.float16\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)", "bias=None"], "tensors": [{"device": "cuda:4", "dtype": "float16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 1024]}]}]`
   - kwargs: `{}`
6. label=`random_low`, calls=`364`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)\n    )\n  arg[1]=Tensor(\n      shape=(1, 1024)\n      dtype=torch.float16\n      device=cuda:5\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=1024, output_features=4096, bias=False, tp_size=8, reduce_results=True)", "bias=None"], "tensors": [{"device": "cuda:5", "dtype": "float16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 1024]}]}]`
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
