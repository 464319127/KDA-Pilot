# KDA-Pilot：LLM Kernel 接口任务

[English](README.md) | 简体中文

这个子目录记录了从真实 B200 服务运行中发现的 SGLang LLM Kernel 优化任务，随后
将它们作为独立的单 GPU Kernel 任务进行优化。

当前 LLM 任务的来源是 SGLang 运行时 Kernel API 日志：

```bash
SGLANG_KERNEL_API_LOGLEVEL=3
SGLANG_KERNEL_API_LOGDEST=/path/to/kernel_api_%i.log
```

生成的任务 shape 是 SGLang Kernel 入口的直接 Python 接口参数及返回值元数据，
而不是 torch-profiler 的 CPU 算子上下文行。

对于每次模型运行，捕获矩阵固定服务命令，并覆盖两种数据集、三个并发级别：

- `random_low`、`random_mid`、`random_high`
- `sharegpt_low`、`sharegpt_mid`、`sharegpt_high`

每个生成的 Kernel 任务包含：

```text
prompt.md       Agent 任务卡
config.toml     任务/构建/基准测试默认配置
baseline/       复制的上游 SGLang 基线源码
solution/       优化后的候选实现源码
bench/          独立基准测试和正确性测试框架
docs/           evidence.json 和来源说明
profile/        供后续优化循环使用的可选性能分析说明
ncu/            可选的 Nsight Compute 报告
tests/          任务本地正确性测试
```

任务目录名是完整 Python Kernel 接口的 slugified 形式，其中点号和符号会转换为
下划线，例如 `sgl_kernel_build_tree_kernel_efficient`。

最重要的原则是对称性：将相关的上游 SGLang 实现复制到 `baseline/`，通过匹配的
本地接口暴露基线和候选实现，并且只在一张空闲的目标 GPU 上测试任务本地代码。
真实 SGLang 服务仅用于发现 shape 和选择目标，不作为正确性或基准测试的基线。
Kernel 优化循环不应要求启动 `sglang serve`、运行 `run_capture`、使用 TP/EP，
也不应要求所有 GPU 都空闲。
