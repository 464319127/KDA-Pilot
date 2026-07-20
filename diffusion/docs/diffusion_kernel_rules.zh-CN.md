# 扩散模型 Kernel 优化规则

[English](diffusion_kernel_rules.md) | 简体中文

这些规则适用于所有独立扩散模型 Kernel 任务。它们恢复了原始长任务 prompt 中的
重要约束，同时让每个任务的 prompt 保持简短。

## 基线与候选配对

每个任务最终必须包含两个本地实现：

- `baseline/`：复制的上游 SGLang Kernel 源码，以及通过任务基准测试 ABI 暴露的本地
  可调用实现。
- `solution/`：通过完全相同的任务基准测试 ABI 暴露的优化实现。

复制基线代码之前，解析上游 SGLang `main` 的最新提交，并使用该确切提交中的扩散
模型 Kernel 源码。不要使用过时的固定 SGLang 提交、状态不明的本地生产 checkout，
或从其他 KDA-Pilot 任务复制的 Kernel 代码。在 `docs/baseline_source.md` 中记录
SGLang 仓库 URL、分支（`main`）、解析到的提交 SHA、解析时间和复制的文件列表。

如果 SGLang 基线 Kernel 是 CUDA/CUDA C++，复制的基线和优化候选必须使用相同的本地
registration/export/build 方式。例如，不要通过一个 Python/JIT 封装暴露基线，却通过
更轻的直接 CUDA 路径暴露候选。如果一侧使用直接的
`TVM_FFI_DLL_EXPORT_TYPED_FUNC` 注册并将输出 Tensor 作为末尾参数传入，另一侧也必须
使用相同模式。

如果复制的 SGLang 实现是 Triton、CuTe DSL 或 Python，应将其保留在 `baseline/` 中，
并构建一个与候选适配器具有相同调用签名、参数顺序、stream 行为和输出分配策略的
本地适配器。

每次 CUDA 启动都必须使用 PyTorch 当前 stream，例如
`at::cuda::getCurrentCUDAStream()`。

## 编译参数

只要可能影响数值或代码生成，基线和候选之间的编译参数就必须对称。

除非复制的上游 SGLang Kernel 已使用 `--use_fast_math` 且候选使用完全相同的参数，
否则不要传递该选项。默认不使用 fast math。

不要只给一侧添加额外的 `nvcc` 参数、架构特定开关或数学模式参数。在
`docs/benchmark_method.md` 中记录所有编译参数。

避免比较不同的构建路径。特别是，不要让一侧使用 `torch.utils.cpp_extension`，另一侧
使用直接本地注册，除非两侧都重新构建并通过等价的封装路径计时。

如果复制的 SGLang Kernel 有相应的 PDL 路径，可以测试 PDL，但它是可选项，并且只有
在任务真实生产工作负载上获胜时才能保留。

## 远程 GPU 规则

B200 任务必须在 B200 上完成正确性验证和基准测试。H200 任务必须在 H200 上完成。

在进行 GPU 工作前，检查 `nvidia-smi`，选择没有活动计算进程且显存没有明显占用的
GPU。在当前运行中，基线、候选、正确性测试、基准测试、性能分析和 NCU 命令都必须
一致使用所选 GPU。

在任务的 `docs/run_log.md` 或 `docs/results.md` 中记录主机、GPU ID、GPU 型号，以及
运行前后的 GPU 状态。

使用任务专属的远程工作区保存构建产物、基准测试日志、性能分析 trace 和 NCU 报告。
不要将产物写入其他任务的工作区。

## 先正确，再性能

开始优化前，恢复以下信息：

- 上游 SGLang 基线源码文件；
- 公开可调用函数的参数和标量参数；
- `docs/diffusion_benchmark_shape_coverage.md` 中的生产工作负载行；
- `docs/diffusion_correctness_contract.md` 中的标准回归网格。

候选实现必须先通过生产工作负载正确性检查和标准回归网格，之后产生的基准测试结果
才算有效。

保留明确的 NaN/Inf 检查。除非任务在 `docs/benchmark_method.md` 中记录了更严格的
任务专属容差，否则使用 `docs/diffusion_correctness_contract.md` 中的容差。

## 基准测试与证据

以 `docs/standalone_diffusion_benchmark_template.py` 作为计时框架起点。调优开始后，
除非基线和候选都重新测量，否则不要修改工作负载、容差、分数聚合方式或计时规则。

每次 RLCR 迭代都必须在选择下一次编辑、基准测试、性能分析或 no-go 结论之前刷新
Kernel 优化上下文。该刷新包括本文档、任务 prompt、当前基准测试证据、
`external/KernelWiki/SKILL.md`，以及可用时的 `external/ncu-report-skill/SKILL.md`。

需要使用 NCU 性能分析时，遵循 `external/ncu-report-skill/SKILL.md`。将性能分析框架、
报告、分析和摘要保存在任务专属目录，并使用产生的证据选择下一次编辑，而不是凭猜测
进行优化。

最终性能声明必须报告：

- 每个工作负载的 median、mean、std、min、p10 和 p90 延迟；
- 生产工作负载上的等权几何平均加速比；
- 完整的命令行；
- 基线源码提交和候选源码 hash；
- GPU 主机/ID/型号和空闲状态证据。

当正确的候选实现尚未明显达到目标，或者性能分析证据可能改变下一次编辑时，应使用
Nsight Compute。最终的改进结论或 no-go 结论必须包含 roofline 风格的说明：估算的
字节移动量、有用的标量或向量操作、实际带宽和/或 FLOP/s（如果相关），以及当前的
性能上限或阻塞因素。

不要因为第一个候选实现落后就直接得出 no-go。no-go 必须包含基线数据、至少一次有
理由的候选尝试、正确性状态、基准测试证据，以及明确命名的当前性能上限或阻塞因素。

## PR 范围

Kernel 优化完成后，最终 PR 只应包含：

- 复制的基线、优化方案、本地 ABI、基准测试适配器以及正确性/基准测试框架所需的
  Kernel 相关源码；
- 按 shape 列出的基线与候选性能对比及最终结论，通常写在 `docs/results.md` 中；
- 为复现结果所需的简短方法和来源说明。

除非用户明确要求将其加入 PR，否则不要提交中间优化产物，例如原始 NCU 报告、Nsight
trace、性能分析运行目录、临时框架二进制文件、构建产物、临时日志、失败实验转储或
大型 benchmark JSONL 文件。将这些产物保留在任务本地或远程工作区用于审计/调试，
然后在创建 PR 前保持未暂存状态。

## Shape 特化

当基准测试或性能分析证据表明不同工作负载分组需要不同的 block size、向量宽度、
内存布局或寄存器压力权衡时，可以使用 shape 特化 Kernel、模板变体、autotune 表和
dispatcher。

使用特化时，在 `docs/dispatch.md` 中记录：

- 分组条件；
- 选中的基线和候选入口；
- 每个分组的延迟和加速比；
- 该分组使用此实现的原因。

如果证据表明多个 shape 分组需要不同实现，不要强行使用一个通用 Kernel。

## 既有工作与探索

在任何 RLCR 迭代中确定实现策略之前，如果 `external/KernelWiki/SKILL.md` 可用，先
阅读或查询它；然后检查可能改变设计的相关上游代码或知识来源：SGLang、CUTLASS/CuTe、
CUDA samples、PyTorch、vLLM、TensorRT-LLM、FlashInfer、DeepGEMM、KernelWiki，以及
任务本地的 NCU 证据。

在 `docs/draft.md`、`docs/results.md` 或 `docs/research.md` 中记录保留和拒绝的想法。
让优化尝试保持有界，并以证据为依据。

## 完成标准

扩散模型任务只有在满足以下条件后才算完成：

- `baseline/`、`solution/`、`bench/` 和 `docs/` 包含所需的本地产物；
- 生产工作负载正确性测试通过；
- 标准回归正确性测试通过；
- 基准测试结果使用标准的独立计时规则；
- NCU 或清晰的 roofline 风格分析解释最终结果或阻塞因素；
- `docs/results.md` 总结最终命令、按 shape 的性能对比、结果和结论；
- 已暂存的 PR diff 不包含原始性能分析、NCU、临时构建和临时调试产物。
