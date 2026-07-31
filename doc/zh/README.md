> [English](../README.md) | 中文

# lnn 文档

面向工程师的 `lnn` 库使用指南。各包的 API 细节在 godoc 里（`go doc github.com/qorm/LNN/tensor`、`go doc github.com/qorm/LNN/autograd`、`go doc github.com/qorm/LNN/nn`、`go doc github.com/qorm/LNN/serialize`）——按 Go 社区惯例，godoc 注释保持英文；这些指南则覆盖签名背后的概念、约定与锋芒之处，中英双语对照。

## 指南

| 指南 | 一句话介绍 |
|---|---|
| [training.md](training.md) | 本库为之设计的训练循环——手写（理解的基础）与 `optimizer` 包（SGD/Momentum/Adam，推荐的生产形态）：参数聚合、ZeroGrad/Backward 纪律、超参热改、指针键状态，以及梯度裁剪为何重要。 |
| [persistence.md](persistence.md) | 逐字节的 `"LNNS"` 线上格式、六个 Save/Load API、优化器状态持久化（`optimizer.SaveState`/`LoadState`，`"LNO1"` 状态流——续训与不间断训练逐位一致）、可运行的「训练 → 保存 → 加载 → 续训」示例，以及不可信流安全契约（先校验后分配、固定限额、渐进分配）。 |
| [shapes-and-broadcasting.md](shapes-and-broadcasting.md) | 完整的广播规则表、（坦诚地说并不对称的）归约形状约定，以及所有输出形状的速查表。 |
| [ltc.md](ltc.md) | 液态时间常数细胞，逐式对照代码：ODE、半隐式欧拉（semi-implicit Euler）代数推导、稀疏突触收缩（sparse contraction）、参数表、`ts` 契约与接线。 |
| [cfc.md](cfc.md) | 闭式连续时间（Closed-form Continuous-time）细胞（Nature Machine Intelligence 2022）：Lemma 1 闭式解（closed-form solution）、Algorithm 1「LTC 编译为闭式」、exprel 稳定化，以及与 LTC 的关系——同一 ODE，解析积分器（analytical integrator）取代欧拉循环。 |
| [architecture.md](architecture.md) | 三层设计（tensor → autograd → nn）外加 optimizer 与 serialize 包、张量为何没有 stride、计算图按算子种类标签派发的反向传播与融合梯度循环如何工作。 |
| [pitfalls.md](pitfalls.md) | 红队审计得出的已知边界与残余风险，以用户须知形式呈现：并发、float32 溢出、重复 Backward、微小 `ts`、不可信模型文件，以及技术债路线图。 |

## 建议阅读顺序

1. **[training.md](training.md)** —— 让模型学起来（先手写循环，后 `optimizer` 包）；其余都是参考。
2. **[persistence.md](persistence.md)** —— 把训练所得存下来，再深入细胞理论。
3. **[ltc.md](ltc.md)** —— 如果你是为液态神经网络而来。
4. **[cfc.md](cfc.md)** —— 闭式兄弟细胞；紧接 ltc.md 阅读，它复用的正是后者的 ODE。
5. **[shapes-and-broadcasting.md](shapes-and-broadcasting.md)** —— 你用库一小时内就会碰到的约定。
6. **[architecture.md](architecture.md)** —— 调试与扩展所需的心智模型。
7. **[pitfalls.md](pitfalls.md)** —— 上线之前必读。

仓库根目录的 `README.md`（中文版 `README_zh.md`）有快速上手示例；`examples/ltc-sequence` 与 `examples/cfc-sequence` 是同一玩具序列任务上两个完整、可运行的训练循环（分别为手写 SGD 与 optimizer 包范式）。示例是仓库的一部分：克隆仓库（`git clone https://github.com/qorm/LNN.git`）后在仓库根目录运行。
