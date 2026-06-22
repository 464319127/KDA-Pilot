# KDA Prompt: sglang_quant_method_model_opt_fp4_linear_method_apply

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `sglang.quant_method.ModelOptFp4LinearMethod.apply`

Goal: optimize or replace this interface for the nvidia/DeepSeek-V3.2-NVFP4 serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `nvidia/DeepSeek-V3.2-NVFP4`
- Model folder: `llm/deepseek_v3_2/b200`
- Category: `quantization`
- Python interface: `sglang.quant_method.ModelOptFp4LinearMethod.apply`
- Captured call count: `28000`
- Captured variants: `512`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `[0]: shape=[1, 256], dtype=uint8, device=cuda:0, contiguous=True`
- `[0]: shape=[1, 256], dtype=uint8, device=cuda:1, contiguous=True`
- `[0]: shape=[1, 256], dtype=uint8, device=cuda:2, contiguous=True`
- `[0]: shape=[1, 256], dtype=uint8, device=cuda:3, contiguous=True`
- `[0]: shape=[10139, 256], dtype=uint8, device=cuda:0, contiguous=True`
- `[0]: shape=[10139, 256], dtype=uint8, device=cuda:1, contiguous=True`
- `[0]: shape=[10139, 256], dtype=uint8, device=cuda:2, contiguous=True`
- `[0]: shape=[10139, 256], dtype=uint8, device=cuda:3, contiguous=True`
- `[0]: shape=[103, 256], dtype=uint8, device=cuda:0, contiguous=True`
- `[0]: shape=[103, 256], dtype=uint8, device=cuda:1, contiguous=True`
- `[0]: shape=[103, 256], dtype=uint8, device=cuda:2, contiguous=True`
- `[0]: shape=[103, 256], dtype=uint8, device=cuda:3, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`244`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=4096, output_features=7168, bias=False, tp_size=4, reduce_results=False)\n    )\n  arg[1]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=4096, output_features=7168, bias=False, tp_size=4, reduce_results=False)", "bias=None"], "tensors": [{"device": "cuda:0", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 4096]}]}]`
   - kwargs: `{}`
2. label=`random_low`, calls=`244`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=4096, output_features=7168, bias=False, tp_size=4, reduce_results=False)\n    )\n  arg[1]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=4096, output_features=7168, bias=False, tp_size=4, reduce_results=False)", "bias=None"], "tensors": [{"device": "cuda:1", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 4096]}]}]`
   - kwargs: `{}`
3. label=`random_low`, calls=`244`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=4096, output_features=7168, bias=False, tp_size=4, reduce_results=False)\n    )\n  arg[1]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=4096, output_features=7168, bias=False, tp_size=4, reduce_results=False)", "bias=None"], "tensors": [{"device": "cuda:2", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 4096]}]}]`
   - kwargs: `{}`
4. label=`random_low`, calls=`244`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=4096, output_features=7168, bias=False, tp_size=4, reduce_results=False)\n    )\n  arg[1]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=4096, output_features=7168, bias=False, tp_size=4, reduce_results=False)", "bias=None"], "tensors": [{"device": "cuda:3", "dtype": "bfloat16", "is_contiguous": true, "kind": "tensor", "name": "arg[1]", "requires_grad": false, "shape": [1, 4096]}]}]`
   - kwargs: `{}`
5. label=`random_low`, calls=`232`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=512, output_features=7168, bias=False, tp_size=4, reduce_results=False)\n    )\n  arg[1]=(\n      [0] Tensor(\n            shape=(1, 256)\n            dtype=torch.uint8\n            device=cuda:0\n            requires_grad=False\n            is_contiguous=True\n          )\n      [1] Tensor(\n            shape=(128, 32)\n            dtype=torch.float8_e4m3fn\n            device=cuda:0\n            requires_grad=False\n            is_contiguous=True\n          )\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=512, output_features=7168, bias=False, tp_size=4, reduce_results=False)", "arg[1]=(", "bias=None"], "tensors": [{"device": "cuda:0", "dtype": "uint8", "is_contiguous"...`
   - kwargs: `{}`
6. label=`random_low`, calls=`232`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=RowParallelLinear(\n      repr=RowParallelLinear(input_features=512, output_features=7168, bias=False, tp_size=4, reduce_results=False)\n    )\n  arg[1]=(\n      [0] Tensor(\n            shape=(1, 256)\n            dtype=torch.uint8\n            device=cuda:1\n            requires_grad=False\n            is_contiguous=True\n          )\n      [1] Tensor(\n            shape=(128, 32)\n            dtype=torch.float8_e4m3fn\n            device=cuda:1\n            requires_grad=False\n            is_contiguous=True\n          )\n    )\nKeyword input arguments:\n  bias=None", "scalars": ["arg[0]=RowParallelLinear(", "repr=RowParallelLinear(input_features=512, output_features=7168, bias=False, tp_size=4, reduce_results=False)", "arg[1]=(", "bias=None"], "tensors": [{"device": "cuda:1", "dtype": "uint8", "is_contiguous"...`
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
