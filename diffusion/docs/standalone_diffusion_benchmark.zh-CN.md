# 独立扩散模型基准测试约定

[English](standalone_diffusion_benchmark.md) | 简体中文

本仓库优化 SGLang 扩散模型 Kernel，但在基准测试运行时不会与 SGLang checkout
交互。SGLang 只作为本地基线的源码提供方。

## 硬性规则

- 在正确性测试或基准测试运行期间，不得修改、导入、monkey-patch 或安装到 SGLang。
- 在实现候选方案之前，将相关的上游 SGLang Kernel 源码复制到 `baseline/`。恢复基线
  时解析上游 SGLang `main` 的最新提交，从该确切提交复制 Kernel 代码，并在
  `docs/baseline_source.md` 中记录上游仓库 URL、分支（`main`）、解析到的提交、解析
  时间和复制的文件列表。
- 基线和候选必须暴露匹配的本地入口。包含在一侧的任何封装开销，也必须包含在另一侧。
- 每个任务必须包含两个本地实现：`baseline/` 中复制的 SGLang 源码，以及
  `solution/` 中的优化实现。
- 两侧都优先使用本地直接 CUDA ABI：`TVM_FFI_DLL_EXPORT_TYPED_FUNC`、
  `tvm::ffi::TensorView` 参数、将输出 Tensor 作为末尾参数传入，以及
  `destination_passing_style = true`。
- 每次 CUDA 启动都必须使用 PyTorch 当前 stream：
  `at::cuda::getCurrentCUDAStream()`。
- 如果复制的 SGLang 实现是 CUDA/CUDA C++，必须让基线和候选使用相同的本地
  registration/export/build 方式。不要通过更重的封装测试复制的 CUDA 基线，却通过
  更轻的直接路径测试候选。
- 不要传递 `--use_fast_math`，除非复制的上游基线已经使用它，并且候选使用完全相同的
  编译参数。
- 如果复制的 SGLang 实现是 Triton、CuTe DSL 或 Python，应将其保留在本地，并使用与
  候选相同的基准测试 ABI 构建本地基线适配器。基准测试不得把重量级 Python 封装与
  轻量 CUDA 封装进行比较。
- 在调优开始前冻结工作负载。修改工作负载、容差、评分或基准测试计时规则后，必须
  删除旧结果，并同时重新测量基线和候选。
- 在任何优化工作开始前，必须根据
  `docs/diffusion_benchmark_shape_coverage.md` 和当前
  `sglang-diffusion-benchmark-profile/benchmark-and-profile.md` 中的 preset，审计扩散
  模型工作负载。

## 首个 Agent 里程碑后的目录内容

```text
baseline/
  复制的上游源码文件
  暴露基线 ABI 的 kernel.cu 或 binding.py
solution/
  暴露候选 ABI 的 kernel.cu 或 binding.py
bench/
  workloads.json
  benchmark.py
  adapter.py
  correctness.py
  results.jsonl
docs/
  baseline_source.md
  benchmark_method.md
  run_log.md
config.toml
```

`bench/benchmark.py` 必须从
[`standalone_diffusion_benchmark_template.py`](standalone_diffusion_benchmark_template.py)
开始。除非该模板存在已记录的 Bug，否则不要另行发明计时框架；如果修复了模板，
必须在修复后重新测量基线和候选。

## ABI 模式

对于纯 CUDA，使用本地直接符号 CUDA 模式：

```cuda
#include <ATen/cuda/CUDAContext.h>
#include <tvm/ffi/container/tensor.h>

void my_kernel(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // launch <<<grid, block, shmem, stream>>>
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, my_kernel);
```

任务可以为上游公开入口分别导出一个函数，也可以导出一个带显式 selector 参数的
函数。基线和候选必须使用相同的选择。

## 工作负载规则

- `bench/workloads.json` 是唯一事实来源。
- 在开始调优前，必须根据 `docs/diffusion_benchmark_shape_coverage.md` 中的目标任务
  系列填充 `bench/workloads.json`。
- 包含任务预期优化的每一个生产 shape，并从
  `docs/diffusion_correctness_contract.md` 添加一组覆盖边界布局/数据类型的小型回归网格。
- 如果当前 SGLang 扩散模型基准 preset 没有对应的保留 shape 行，则必须添加新的实时
  捕获行，或在 `docs/benchmark_preset_audit.md` 中记录实时“无调用”证明。
- 每个工作负载记录函数/selector、Tensor shape、数据类型、stride、标量参数、容差、
  随机种子，以及是否计入核心评分。
- 不得静默跳过生产工作负载。任何基线缺失、候选缺失、编译失败、运行时失败或正确性
  失败，都会使基准测试无效。

## 计时规则

- 尽可能在隔离的子进程中运行每个工作负载。
- 为每次试验生成新的随机输入；在一次试验内部可以复用输入，但不同试验之间必须更换。
- 在计时前预分配输出 Tensor。计时区域不得包含输入生成、Python 准备、JIT 构建、导入、
  内存分配或数据恢复。
- 预热基线和候选两侧。
- 使用 CUDA Event。包含封装在内的 wall-clock 时间只能作为辅助诊断指标。
- 使用内循环放大：在一对 Event 之间连续调用 N 次，然后除以 N。逐步增加 N，直到
  Event 样本达到约 1000 us，或 N 达到配置上限。
- 每次试验交错进行 A/B 采样，以抵消时钟和温度变化：
  baseline、candidate、baseline、candidate，或者由确定性种子选择相反顺序。
- 为每个工作负载的两侧报告 median、mean、std、min、p10、p90。
- 每个工作负载的主要加速比为 `baseline_median_us / candidate_median_us`。
- 核心指标是所有生产工作负载的等权几何平均。同时将算术平均作为辅助跟踪指标。

推荐默认值：

```toml
[benchmark]
warmup_runs = 10
iterations = 200
num_trials = 7
inner_iterations_min = 1
inner_iterations_max = 4096
target_sample_us = 1000
timeout_seconds = 600
use_isolated_runner = true
```

## 标准基准测试文件

使用 [`standalone_diffusion_benchmark_template.py`](standalone_diffusion_benchmark_template.py)
作为 `bench/benchmark.py` 的必需起点。

模板固定了以下基准测试策略：

- 每个工作负载都可以在隔离子进程中运行；
- 每次试验接收新的随机输入，同时在试验内部保持 Tensor 对象稳定；
- 基线和候选输出在计时之外预先分配；
- 计时前运行正确性测试，并对输出缓冲区执行 poison；
- 基线/候选在每次试验中按确定性顺序交错计时；
- CUDA Event 提供主要 GPU 时间，wall-clock 样本作为诊断信息；
- 按两侧分别校准内循环放大，直到 Event 样本足够长；
- 每个工作负载输出 median/mean/std/min/p10/p90 和原始样本；
- 核心评分是生产工作负载上的等权几何平均；
- 结果 JSONL 记录命令、环境、GPU 状态和基准测试设置。

任务专属的 `bench/adapter.py` 只负责创建 Tensor 和调用两个 ABI：

```python
def make_case(workload, *, device, seed):
    ...

def call_baseline(workload, inputs, outputs):
    ...

def call_candidate(workload, inputs, outputs):
    ...
```

`call_baseline` 和 `call_candidate` 必须暴露完全相同的封装开销。计时路径中不得分配
输出 Tensor。

## 正确性规则

- 在可行时，将候选和基线与独立的 PyTorch/数学 oracle 进行比较。如果完整 oracle
  成本过高，至少要将候选与复制的基线以及选定的 oracle 行进行比较。
- 检查每个输出的 shape、数据类型、NaN/Inf 和容差。
- 每次正确性测试前对输出缓冲区执行 poison，以便发现过期输出和跳过 Kernel 的 Bug。
- 如果使用 CUDA Graph capture，添加零输出 replay、poison-cell 和变化输入测试，证明
  Kernel 确实完成了 replay。

## 来源记录

每条基准测试结果都必须记录：

- 任务 slug 和目标 GPU；
- 上游基线提交及复制的文件；
- 候选源码 hash；
- 完整命令；
- CUDA、PyTorch、编译器和 TVM-FFI 版本；
- GPU 型号、GPU ID，以及运行前后的空闲状态；
- 工作负载数量及 trial/iteration/inner-loop 设置；
- 正确性摘要。

没有这些来源信息时，不要保留基准测试数字。
