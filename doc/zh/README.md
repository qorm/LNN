> [English](../README.md) | 中文

# LNN 文档

面向工程师的 `LNN` 库使用指南。各包的 API 细节在 godoc 里（`go doc github.com/qorm/LNN/tensor`、`go doc github.com/qorm/LNN/autograd`、`go doc github.com/qorm/LNN/nn`、`go doc github.com/qorm/LNN/optimizer`、`go doc github.com/qorm/LNN/serialize`）——按 Go 社区惯例，godoc 注释保持英文；这些指南则覆盖签名背后的概念、约定与锋芒之处，中英双语对照。

文档有两条轴：**概念指南**（库是什么、为何如此）与**任务文档**
（怎么做某件具体的事）。带着"我想做 X"来的话，从
[cookbook.md](cookbook.md) 或 [faq.md](faq.md) 进入；想理解引擎
本身，则按下文的阅读顺序。

## 指南

| 指南 | 一句话介绍 |
|---|---|
| [training.md](training.md) | 本库为之设计的训练循环——手写（理解的基础）与 `optimizer` 包（SGD/Momentum/Adam/AdEMAMix/Schedule-Free AdamW，推荐的生产形态）：参数聚合、ZeroGrad/Backward 纪律、超参热改、指针键状态，以及梯度裁剪为何重要。 |
| [persistence.md](persistence.md) | 逐字节的 `"LNNS"` 线上格式、六个 Save/Load API、优化器状态持久化（`optimizer.SaveState`/`LoadState`，`"LNO1"` 状态流——续训与不间断训练逐位一致）、可运行的「训练 → 保存 → 加载 → 续训」示例，以及不可信流安全契约（先校验后分配、固定限额、渐进分配）。 |
| [shapes-and-broadcasting.md](shapes-and-broadcasting.md) | 完整的广播规则表、（坦诚地说并不对称的）归约形状约定，以及所有输出形状的速查表。 |
| [ltc.md](ltc.md) | 液态时间常数细胞，逐式对照代码：ODE、半隐式欧拉（semi-implicit Euler）代数推导、稀疏突触收缩（sparse contraction）、参数表、`ts` 契约与接线。 |
| [cfc.md](cfc.md) | 闭式连续时间（Closed-form Continuous-time）细胞（Nature Machine Intelligence 2022）：Lemma 1 闭式解（closed-form solution）、Algorithm 1「LTC 编译为闭式」、exprel 稳定化，以及与 LTC 的关系——同一 ODE，解析积分器（analytical integrator）取代欧拉循环。 |
| [architecture.md](architecture.md) | 三层设计（tensor → autograd → nn）外加 optimizer 与 serialize 包、张量为何没有 stride、计算图按算子种类标签派发的反向传播与融合梯度循环如何工作。 |
| [pitfalls.md](pitfalls.md) | 红队审计得出的已知边界与残余风险，以用户须知形式呈现：并发、float32 溢出、重复 Backward、微小 `ts`、不可信模型文件，以及技术债路线图。 |
| [cookbook.md](cookbook.md) | 任务式食谱集，每条都是完整的实测程序：最小回路、Adam 加裁剪、梯度累积、逐位一致的断点续训、变 `ts` 事件驱动序列、模型检查、自定义损失、LTC 与 CfC 选型、多模块组合、不可信文件加载、学习率退火、确定性复现、`UnrollRemat` 长序列训练（分块 BPTT）、Schedule-Free AdamW 与 train/eval 契约。 |
| [faq.md](faq.md) | 常见问题，简短回答：loss 不降、`NaN` 损失、`ts`/`units`/`unfolds` 选取、梯度累积语义、跨平台末位差异、加载报错解读、Adam 续训。 |
| [api.md](api.md) | API 速查：各包导出符号的一行式名录，并指向其权威文档（godoc / 概念指南）。（新增——由并行的文档任务创建。） |
| [pgo.md](pgo.md) | PGO（画像引导优化）的诚实实测：为什么库本身不附带画像、给自己的二进制采画像的工作流，以及本仓库的基准数字——包括头条收益所悬的那个双态内联决策。 |

## 按读者画像选路

**① 第一次用 LNN。** 根目录 [`README.md`](../../README.md)（中文版
[`README_zh.md`](../../README_zh.md)）的快速上手（可复制即运行）→
[training.md](training.md)（循环及其纪律）→ [cookbook.md](cookbook.md)
前三条食谱（最小回路、optimizer 加裁剪、梯度累积）。读完即可开始
训练。

**② 要部署 / 做检查点。** [persistence.md](persistence.md)（格式与
六个 Save/Load API）→ [cookbook.md](cookbook.md) 食谱 4（训练 →
保存 → 加载 → 续训，逐位一致，含 Adam 状态）→ [faq.md](faq.md)
"加载报错怎么读" → 加载陌生人的文件之前先读 [pitfalls.md](pitfalls.md)
§10。

**③ 从 ncps / PyTorch 迁移。** [ltc.md](ltc.md) 与 [cfc.md](cfc.md)
（「与 ncps 参考实现的关系」表把每个 ncps 概念映射到 LNN 对应物
——`elapsed_time` → `ts`、`implicit_param_constraints` → softplus、
默认区间逐字采用）→ [shapes-and-broadcasting.md](shapes-and-broadcasting.md)
（广播是*枚举子集*，不是 NumPy 的；归约形状不对称——迁移者最先
被咬到的两个约定）→ [pitfalls.md](pitfalls.md)（契约式单线程、
调用方负责的裁剪、没有框架层——哪些东西是刻意不做的）。

**④ 审计内部 / 参与贡献。** [architecture.md](architecture.md)
（三层与图的机理）→ 各包 godoc（`go doc github.com/qorm/LNN/tensor`、
`…/autograd`、`…/nn`、`…/optimizer`、`…/serialize`）→ 仓库的
`PLAN.md` 与 `PROGRESS.md`（完整的开发与红队审计轨迹，逐阶段
记录）→ [pitfalls.md](pitfalls.md) 的技术债路线图，了解哪些是
已知且接受的。

## 建议阅读顺序

1. **[training.md](training.md)** —— 让模型学起来（先手写循环，后 `optimizer` 包）；其余都是参考。
2. **[persistence.md](persistence.md)** —— 把训练所得存下来，再深入细胞理论。
3. **[ltc.md](ltc.md)** —— 如果你是为液态神经网络而来。
4. **[cfc.md](cfc.md)** —— 闭式兄弟细胞；紧接 ltc.md 阅读，它复用的正是后者的 ODE。
5. **[shapes-and-broadcasting.md](shapes-and-broadcasting.md)** —— 你用库一小时内就会碰到的约定。
6. **[architecture.md](architecture.md)** —— 调试与扩展所需的心智模型。
7. **[pitfalls.md](pitfalls.md)** —— 上线之前必读。

任务文档横切这一顺序：[cookbook.md](cookbook.md) 与 [faq.md](faq.md)
的每条食谱、每个回答都从其*原理*处链回相应的概念指南，因此可以
从它们任何一点进入（上文的「按读者画像选路」也按画像给出了路由）。

仓库根目录的 `README.md`（中文版 `README_zh.md`）有快速上手示例；`examples/ltc-sequence` 与 `examples/cfc-sequence` 是同一玩具序列任务上两个完整、可运行的训练循环（分别为手写 SGD 与 optimizer 包范式）。示例是仓库的一部分：克隆仓库（`git clone https://github.com/qorm/LNN.git`）后在仓库根目录运行。
