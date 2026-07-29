> [English](../README.md) | 中文

# lnn 文档

面向工程师的 `lnn` 库使用指南。各包的 API 细节在 godoc 里（`go doc lnn/tensor`、`go doc lnn/autograd`、`go doc lnn/nn`）——按 Go 社区惯例，godoc 注释保持英文；这些指南则覆盖签名背后的概念、约定与锋芒之处，中英双语对照。

## 指南

| 指南 | 一句话介绍 |
|---|---|
| [training.md](training.md) | 如何编写本库为之设计的手写训练循环：参数聚合、ZeroGrad/Backward 纪律、朴素 SGD，以及梯度裁剪为何重要。 |
| [shapes-and-broadcasting.md](shapes-and-broadcasting.md) | 完整的广播规则表、（坦诚地说并不对称的）归约形状约定，以及所有输出形状的速查表。 |
| [ltc.md](ltc.md) | 液态时间常数细胞，逐式对照代码：ODE、半隐式欧拉（semi-implicit Euler）代数推导、参数表、`ts` 契约与接线。 |
| [architecture.md](architecture.md) | 三层设计（tensor → autograd → nn）、张量为何没有 stride、计算图及其反向闭包如何工作。 |
| [pitfalls.md](pitfalls.md) | 红队审计得出的已知边界与残余风险，以用户须知形式呈现：并发、float32 溢出、重复 Backward、微小 `ts`，以及路线图。 |

## 建议阅读顺序

1. **[training.md](training.md)** —— 让模型学起来；其余都是参考。
2. **[shapes-and-broadcasting.md](shapes-and-broadcasting.md)** —— 你用库一小时内就会碰到的约定。
3. **[ltc.md](ltc.md)** —— 如果你是为液态神经网络而来。
4. **[architecture.md](architecture.md)** —— 调试与扩展所需的心智模型。
5. **[pitfalls.md](pitfalls.md)** —— 上线之前必读。

仓库根目录的 `README.md`（中文版 `README_zh.md`）有快速上手示例；`examples/ltc-sequence` 是一个完整、可运行的训练循环，任务是一个玩具序列任务。
