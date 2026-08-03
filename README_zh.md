> [English](README.md) | 中文

# LNN

[![CI](https://github.com/qorm/LNN/actions/workflows/ci.yml/badge.svg)](https://github.com/qorm/LNN/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/qorm/LNN.svg)](https://pkg.go.dev/github.com/qorm/LNN)

一个纯 Go 数值计算库：稠密 `float32` 张量（tensor）、反向模式自动微分，以及液态时间常数（Liquid Time-Constant，LTC）神经细胞——**零第三方依赖**（唯一的导入就是标准库）。LTC 实现遵循 Hasani 等人的论文
[Liquid Time-constant Networks](https://ojs.aaai.org/index.php/AAAI/article/view/17017)
（AAAI 2021）及参考实现 [`mlech26l/ncps`](https://github.com/mlech26l/ncps)。

LNN 小而显式。它宁可牺牲覆盖面，也要保证内核可读、可审计：没有代码生成，没有 GPU 后端，没有运算符重载技巧——只有 Go。

在当前的文献脉络里，液态细胞正被读作非线性状态空间模型（SSM）：Liquid-S4（[arXiv:2209.12951](https://arxiv.org/abs/2209.12951)，ICLR 2023）把 LTC 动力学装进了 S4 状态空间形式体系，LrcSSM（[arXiv:2505.21717](https://arxiv.org/abs/2505.21717)，NeurIPS 2025）则把液态动力学扩展成能线性时间处理长序列的非线性 SSM 层。从这个角度看，本库的细胞就是可解释、变步长（变 `ts`）、稀疏接线的非线性 SSM 细胞——CfC 更新步与 Mamba 选择性 SSM 步的公式级对应，见 [doc/zh/cfc.md](doc/zh/cfc.md)。

## 包结构

| 包 | 职责 |
|---|---|
| `github.com/qorm/LNN/tensor` | 稠密行主序 `float32` 张量，聚焦 1D/2D 的算子集：矩阵乘、带有限广播（broadcasting）的逐元素运算、激活、归约、切片、随机初始化。自 v0.4.0 起，rank ≤ 4 的形状内联存储于张量本体（每次构造少一次堆分配），`Tensor.Reshape` 可重指形状而不重新分配数据。 |
| `github.com/qorm/LNN/autograd` | 动态计算图（computation graph）引擎。每个算子给其输出 `Variable` 打上算子种类（op kind）标签；`Backward` 按逆拓扑序遍历计算图，派发每个节点的梯度传播，将梯度累加（gradient accumulation）到叶节点（leaf）。 |
| `github.com/qorm/LNN/nn` | 神经网络构件：`Linear` 层、`Wiring` 突触（synapse）拓扑、`LTC` 液态细胞及其闭式（closed-form）兄弟细胞 `CfC`，以及在序列上驱动循环细胞的 `Cell`/`Unroll` 抽象。 |
| `github.com/qorm/LNN/optimizer` | 作用于 `autograd` 的显式参数更新规则：SGD、经典重球动量（momentum）Momentum、Adam（Kingma & Ba，含偏差校正（bias correction））、AdEMAMix（ICLR 2025，双梯度 EMA）、Schedule-Free AdamW（NeurIPS 2024，以迭代平均取代衰减调度，附 Train/Eval 转换契约）。一次 `Step(params)` 调用替换手写更新循环。状态持久化（`SaveState`/`LoadState`，`"LNO1"` 状态流）使续训（resume）与不间断训练逐位一致。 |
| `github.com/qorm/LNN/serialize` | 带版本的二进制持久化：紧凑的小端序张量流（`"LNNS"`，version 2——写 v2 并带 CRC-32C 校验和，仍可读 v1），其加载路径把输入视为不可信——一切失败都是 error（绝不 panic），尺寸声明先校验后分配，未知长度读端渐进分配。它是 `nn` 六个 Save/Load 函数背后的存储层。 |

## 文档

面向库使用者的指南位于 [`doc/`](doc/)，中文版位于 [`doc/zh/`](doc/zh/)：

| 指南 | 内容 |
|---|---|
| [doc/zh/training.md](doc/zh/training.md) | 手写训练循环与 `optimizer` 包（SGD/Momentum/Adam）、梯度裁剪（gradient clipping）、发散排查清单 |
| [doc/zh/persistence.md](doc/zh/persistence.md) | `"LNNS"` 线上格式规格、六个 Save/Load 函数、优化器状态持久化（`"LNO1"` 状态流、逐位续训）、不可信流安全契约、可运行的「训练→保存→加载→续训」示例 |
| [doc/zh/shapes-and-broadcasting.md](doc/zh/shapes-and-broadcasting.md) | 广播规则表、归约输出形状、非对称约定 |
| [doc/zh/ltc.md](doc/zh/ltc.md) | LTC 论文↔代码对照、参数表、`ts` 契约、接线（wiring） |
| [doc/zh/cfc.md](doc/zh/cfc.md) | CfC 闭式细胞：Lemma 1 论文对照、exprel 稳定化、与 LTC 的关系 |
| [doc/zh/architecture.md](doc/zh/architecture.md) | 三层设计、计算图机制、`float32` 约束 |
| [doc/zh/pitfalls.md](doc/zh/pitfalls.md) | 并发契约、溢出场景、残余风险、路线图 |

建议的阅读顺序见 [doc/zh/README.md](doc/zh/README.md)。
各包的 API 参考以 godoc 为准：`go doc github.com/qorm/LNN/tensor`、`go doc github.com/qorm/LNN/autograd`、`go doc github.com/qorm/LNN/nn`。
按 Go 社区惯例，godoc 注释（包括三个 `doc.go`）保持英文；本中文版文档与英文 `doc/` 一一对应，可交叉查阅。

## 安装

模块路径为 `github.com/qorm/LNN`，直接获取：

```
go get github.com/qorm/LNN@latest
```

若要从源码工作，克隆仓库并用 `replace` 指令引用本地检出：

```
git clone https://github.com/qorm/LNN.git
```

```go
// your app's go.mod
module myapp

go 1.26.5

require github.com/qorm/LNN v0.0.0

replace github.com/qorm/LNN => ../LNN
```

在仓库内部，`make build` / `make test` 开箱即用。

## 快速上手

训练循环就是对着 `autograd` 手写的——这是理解本库的基础——而 `optimizer` 包把同一个循环打包成 SGD/Momentum/Adam/AdEMAMix/Schedule-Free AdamW 供生产使用。下面的程序用一个手搓的线性模型配合朴素 SGD 拟合 `y = 2x + 1`，只用到 `tensor` 和 `autograd`，`go run` 即可运行。把手写更新替换为 `optimizer.NewSGD(lr)` + `Step(params)`，输出完全一致——见 [doc/zh/training.md](doc/zh/training.md)。

朴素的 `float32` SGD 没有任何内置稳定化措施：请使用温和的学习率；在稍大的问题上考虑对全局梯度范数做裁剪（`examples/ltc-sequence` 将最大范数裁剪到 1.0）。

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

func main() {
	rng := rand.New(rand.NewSource(42))

	// 训练数据：y = 2x + 1，存成 [n,1] 矩阵。
	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1 // U(-1, 1)
		ys[i] = 2*xs[i] + 1
	}
	x := autograd.Const(tensor.FromData(xs, n, 1))
	y := autograd.Const(tensor.FromData(ys, n, 1))

	// 模型参数：y = x*w + b。
	w := autograd.Var(tensor.Randn(rng, 1, 1))
	b := autograd.Var(tensor.New(1))

	const epochs, lr = 200, 0.1
	for epoch := 0; epoch < epochs; epoch++ {
		// 前向：每轮迭代都构建一张全新的计算图。
		pred := autograd.Add(autograd.MatMul(x, w), b) // [n,1]
		diff := autograd.Sub(pred, y)                  // [n,1]
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))

		// 反向：梯度累加进叶节点 w 和 b。
		w.ZeroGrad()
		b.ZeroGrad()
		loss.Backward()

		// 手写 SGD 更新步。
		for i := range w.Data.Data {
			w.Data.Data[i] -= lr * w.Grad.Data[i]
		}
		for i := range b.Data.Data {
			b.Data.Data[i] -= lr * b.Grad.Data[i]
		}

		if epoch%40 == 0 || epoch == epochs-1 {
			fmt.Printf("epoch %3d  loss=%.6f  w=%.4f  b=%.4f\n",
				epoch, loss.Value(), w.Data.Data[0], b.Data.Data[0])
		}
	}
}
```

实际输出（Go 1.26，seed 42）：

```
epoch   0  loss=1.398637  w=0.5503  b=0.1674
epoch  40  loss=0.006162  w=1.8658  b=0.9795
epoch  80  loss=0.000047  w=1.9883  b=0.9982
epoch 120  loss=0.000000  w=1.9990  b=0.9998
epoch 160  loss=0.000000  w=1.9999  b=1.0000
epoch 199  loss=0.000000  w=2.0000  b=1.0000
```

_输出录制于 arm64（Apple Silicon）；在其他架构上，受 Go 允许的浮点缩合（FMA 融合乘加舍入）影响，末尾数字可能有微小差异。_

这个循环把 `w ≈ 2`、`b ≈ 1` 恢复到了 `float32` 的精度极限。

### 使用 `nn` 包

```go
rng := rand.New(rand.NewSource(1))

// 全连接层：y = x @ W + b。
fc := nn.NewLinear(4, 8, rng)   // W: [4,8], B: [8]（Xavier 均匀初始化）
y := fc.Forward(x)              // x: [batch,4] -> y: [batch,8]
params := fc.Parameters()       // []*autograd.Variable{W, B}

// 液态时间常数细胞。反转电位（reversal potential）是固定的 ±1 常量，
// 不属于 Parameters()。
cell := nn.NewLTC(4, 8, nil, 6, rng) // inDim=4, units=8, 全连接接线, 6 次 ODE 展开（unroll）
out, h := cell.Step(x, nil, 0.1)     // x: [batch,4], nil = 零初始状态, 时间跨度 ts=0.1

// 在序列上展开任意 nn.Cell；整条序列都留在计算图里，
// 因此在 ys 上构建的损失只需一次 Backward 就能对时间求导。
ys, hN := nn.Unroll(cell, xs, nil, 0.1) // xs: []*autograd.Variable，每个为 [batch,4]

// 长序列：UnrollRemat 对 lossFn(ys) 做贯穿时间的求导，
// 梯度与 Unroll + loss.Backward() 逐位相等，
// 峰值图内存为 O(chunkSize) 而非 O(len(xs))。
// params 必须列出细胞 Step 消费的每一个可训练叶（完备性有审计）；
// lossFn 由 detach 后的逐步输出构建标量损失。
ys, hN, loss := nn.UnrollRemat(cell, cell.Parameters(), xs, nil, 0.1, 8, lossFn)

// CfC 是闭式兄弟细胞：同一条 ODE、同一套 13 参数突触参数化，
// 但没有 unfolds——Lemma 1 闭式解一步推进整个时间跨度 ts（doc/zh/cfc.md）。
cfc := nn.NewCfC(4, 8, nil, rng)
out2, h2 := cfc.Step(x, nil, 0.1)

// 自阶段 18 起，这个闭式步也是单个融合图节点——对调用方透明：
// API 不变、前向与梯度逐位一致，本机每步约 34 µs、52 allocs。
```

`ts` 的选取以任务的采样间隔为锚：若序列每一步对应底层过程的一个时间单位，则 `ts = 1.0`。完整的 `ts` 契约——须为正的有限值，`ts ≈ 1e-38` 以下仅为有限性域——见 [doc/zh/ltc.md](doc/zh/ltc.md)。

示例按难度梯度编排——先从 hello-train 与 hello-use 入手（各几十行），再看完整训练循环，最后进入续训与诊断等进阶主题。示例都在仓库内——克隆仓库（`git clone https://github.com/qorm/LNN.git`）后在仓库根目录运行；`go get` 安装的用户可直接在 GitHub 上浏览。任务式中文训练配方（梯度累积、断点续训、自定义损失、事件驱动变 ts 等）详见 [doc/zh/cookbook.md](doc/zh/cookbook.md)。

**入门**

- `examples/hello-train`——最简训练程序：用单个 CfC 神经元拟合 `y = 2x + 1`，更新步是朴素手写梯度下降（约 60 行编号注释：模型、数据、前向 → 反向 → 更新），并把模型保存下来。
- `examples/hello-use`——最简推理程序：加载上面保存的模型，只做一次前向步，把预测与真值直线并排打印（约 30 行，没有任何训练代码）。

**完整训练循环**

- `examples/ltc-sequence`——玩具「带界累加器」序列任务上的完整训练循环（对 `nn.ParametersOf(cell, readout)` 做手写 SGD，含全局梯度范数裁剪）：`go run ./examples/ltc-sequence`。
- `examples/cfc-sequence`——在同一任务上换用 CfC 细胞与推荐的生产形态——调用方负责的裁剪加 `optimizer.NewSGD` + `Step`——损失从 `0.620651` 降到 `0.029091`。

**进阶**

- `examples/ltc-resume`——把训练到一半的 LTC 检查点（模型、读出层参数，以及经 `optimizer.SaveState` 保存的 Adam 状态）写入临时目录，再用另一组种子构建的全新一套对象加载续训，并断言续训轨迹与不间断训练逐位（bit-identical）一致。
- `examples/gradient-inspect`——训练一个小 LTC，每隔 K 轮打印各参数的梯度 L2 范数、全局梯度最大绝对值、NaN/Inf 计数与参数更新量——含同轮「裁剪前 vs 裁剪后」的范数对比——作为定位 loss 停滞或发散的诊断模板。

## 数值与规模

- **全局 `float32`。** 张量数据是行主序的扁平 `[]float32`，没有 `float64` 模式。
- **聚焦 1D/2D。** `MatMul` 只对矩阵定义；`Rows`/`Cols` 作用于非 2D 张量会 panic。逐元素算子支持任意形状。
- **广播是一个显式枚举的子集**——不是通用的 NumPy 式广播。二元逐元素算子恰好接受以下组合：
  - 形状相同；
  - 标量（恰含一个元素的张量）对任意形状；
  - 行向量（`[n]` 或 `[1,n]`）对 `[m,n]` 矩阵；
  - 列向量（`[m,1]`）对 `[m,n]` 矩阵；
  - `[m,1]` 与 `[1,n]`，产出外积 `[m,n]`。

  其他任何组合都会 panic，并附说明性消息。
- **形状约定并非完全对称**（例如 `SumRows` 返回 `[1,n]` 而 `SumCols` 返回 `[m]`，1D⊕1D 的结果会被提升为 `[1,n]`）。依赖某个归约的输出形状之前，请先读 `tensor/ops.go` 里的文档注释。
- **计算图保留到 `Backward` 为止。** 每个中间张量都被计算图持有，因此内存随算子数量增长。一次 LTC step——感知突触与 `unfolds` 轮 ODE 迭代同在内——以**单个融合图节点**运行（任意维度下每步 34 个节点；前向与梯度对旧图路径逐位一致），构建于同一套稀疏突触前收缩（sparse contraction）之上——`+0` 播种、末端归一化 MatMul 收尾的折叠（fold），全程不存在任何稠密 `[units², units]` 指示矩阵（indicator matrix）（`units = 1024` 全接线细胞的构造耗费约 32 MB，而非早期稠密指示矩阵设计所需的约 8 GiB）。历经多轮反向、分配与融合优化，本机实测（`-benchtime=200x`）分配数已降至 `LTCStep` **77 allocs/op**（约 78 µs、23 KB）、`UnrollBackward` **3,750**（约 1.29 ms、558 KB）——较最初的逐突触循环下降约 99%/97%，较 v0.5.2 再降 allocs 67%/44%（阶段 19：感知路径入核、VJP 暂存面复用、`Backward` DFS 暂存池化、广播形状定长数组）。在这个引擎上，`units`、`unfolds` 和序列长度请保持适度；`CfC` 细胞（[doc/zh/cfc.md](doc/zh/cfc.md)）完全没有 `unfolds` 因子，且同样把整步融合为单节点（52 allocs/op，约 34 µs）；而真正长的序列可用 `nn.UnrollRemat` 以 O(chunk) 峰值图内存代替 O(序列) 对时间求导——T = 512 实测保留约 0.65 MB，对比全展开驻留约 8.3 MB（内存语义与最坏情形见 [doc/zh/architecture.md](doc/zh/architecture.md) 与 [doc/zh/pitfalls.md](doc/zh/pitfalls.md)；完整食谱见 [doc/zh/cookbook.md](doc/zh/cookbook.md#13-长序列训练unrollremat-分块-bptt)）。

## 并发契约

**LNN 在设计上是单线程的。**

- `Backward` 会修改叶变量的 `Grad` 缓冲区；在共享参数的变量上并发运行它属于数据竞争（data race），会丢失或损坏梯度（已在 `go test -race` 下实证）。
- `Variable` 和 `Tensor` 直接暴露其存储（`Data`），且不带任何同步机制。不要从多个 goroutine 读写同一个张量。
- 用于初始化的 `math/rand.Rand` 同样不是 goroutine 安全的。

并行负载的受支持模式：给每个 goroutine 它自己的张量、变量和 RNG，绝不跨 goroutine 共享计算图或其参数。

## 状态与路线图

截至本提交的诚实成熟度评估（覆盖率为 `go test -cover` 包内实测）：

| 包 | 状态 |
|---|---|
| `tensor` | 核心稳定、测试充分（约 99.7% 行覆盖率）。唯一残余的未覆盖语句是 `broadcastBinary` 里一处双常量填充循环体，已论证为不可达（该路径上列数恒为 `1`，循环永不执行，且 `[1,1]×[1,1]` 会被同形快路径先截）；列明而非强凑一个造作的测试。转置感知 MatMul 内核由 `autograd` 包的测试覆盖。v0.4.0 的内嵌形状 backing（embedded backing）消除了每次张量构造的一次堆分配（基准 allocs −18~−26%，墙钟在噪声内持平）。 |
| `autograd` | 稳定、测试充分（100% 行覆盖率）；已覆盖路径上的梯度均通过有限差分与逐位差分检验，包括为异形手设梯度新增的旧组合回退分支，以及 Sigmoid–Hadamard 融合的常规与回退路径。 |
| `nn` | 可用、测试充分（约 99.9% 行覆盖率——唯一未覆盖语句是 `UnrollRemat` 单元扫描中一处构造性不可达的 nil-root 防火墙守卫，已在 `nn/remat.go` 注释中论证；与 optimizer 行同一披露纪律）：LTC 与 CfC 的前向/反向路径有回归测试，包括闭式退化情形检验、微小/NaN `ts` 防护、接线校验与 Save/Load round-trip（加载上限处有 units=2048 合法流真实 round-trip）。两个细胞的反转电位都是固定的 ±1 常量，由稀疏收缩（sparse contraction）之上的行视图常量承载——不可训练，也没有死梯度（结构上不可能）。稀疏收缩有对旧指示矩阵（indicator matrix）实现的逐位回归与大型细胞内存门禁。两个细胞的热路径都以融合 `FusedOp` 节点运行（LTC 的感知驱动与 ODE 展开——自阶段 19a 起任意维度下每步 34 个节点，VJP 暂存面每次反向只分配一次；CfC 自阶段 18 起的整个闭式步），各自对融合前图路径在多种细胞形状、损失形状、链式/堆叠拓扑与对抗性非有限输入下逐位回归，并由三个逐位差分 fuzz 目标守门（共 11 个原生 fuzz 目标，CI 在 amd64 与 arm64 双架构门禁）；`UnrollRemat` 的分块 BPTT 以「配置 × 损失形状」差分矩阵对全图反向逐位回归。CfC 是较新的细胞，API 仍可能演进。 |
| `optimizer` | 稳定，约 99.8% 行覆盖率（未覆盖残余为物理不可达的参数计数守卫）：五条更新规则均与独立参考实现对照验证（SGD 逐位一致，Adam 约 1.6e-6，AdEMAMix 约 9.5e-6、ScheduleFreeAdamW 约 3.3e-7——后两者对照各自论文的第三方 float64 参考，带鉴别力守门），指针键状态语义有回归测试；ScheduleFreeAdamW 的 train/eval 模式契约在对 eval 模式参数 Step 时先于任何修改 panic 指名；状态持久化（`SaveState`/`LoadState`，`"LNO1"` 状态流，kind 0–4）以续训逐位等价测试（50+50 vs 100 步，五种优化器全过，含横跨 warmup 边界）与恶意流测试（先全验后应用、零副作用、字节预算门禁）钉住。 |
| `serialize` | 稳定，97.8% 行覆盖率：round-trip 逐位精确性（含 NaN 与 −0）有回归测试，并以提交的黄金向量做字节级钉死；不可信流契约——固定限额先校验后分配（含加载路径 `units`/`inDim` 上限 2048，按稀疏收缩的 O(units²) 加载期内存量级核定）、未知长度读端渐进分配——以分配计数与字节预算测试钉住；变异模糊测试数千个变异体 0 panic，资源耗尽加固后再测一轮依然 0 panic。资源边界文档见 [doc/zh/persistence.md](doc/zh/persistence.md)。 |

CfC（Closed-form Continuous-time）细胞与内置优化器是最新加入的特性：`nn.CfC`（[doc/zh/cfc.md](doc/zh/cfc.md)）API 仍可能演进；`optimizer` 包（SGD/Momentum/Adam/AdEMAMix/Schedule-Free AdamW）与 `serialize` 包加 `nn` 的六个 Save/Load 函数（[doc/zh/persistence.md](doc/zh/persistence.md)）已稳定。手写循环依然有效，也仍是理解引擎的基础。序列展开由通用的 `nn.Unroll` 助手覆盖；`examples/ltc-sequence` 以手写 SGD 展示端到端训练范式，`examples/cfc-sequence` 在同一任务上展示 CfC 细胞加推荐 optimizer 形态（损失 `0.621 → 0.029`）。后续加固在不破坏文档内既有用法的前提下让引擎保持精小：稀疏突触收缩（sparse contraction）消灭 O(units³) 指示矩阵（indicator matrix）（`units = 1024` 构造约 32 MB，而非约 8 GiB），优化器状态持久化（`SaveState`/`LoadState`，`"LNO1"` 状态流）使三优化器的续训逐位等价，一轮 API 卫生收尾把形状内嵌进每个张量（基准 allocs −18~−26%、墙钟持平；新增导出的 `Tensor.Reshape` 取代 `Shape` 直写）、删除无调用方的 `tensor.Stack`、并经评估后冻结非对称归约约定（决策留档见 [doc/zh/shapes-and-broadcasting.md](doc/zh/shapes-and-broadcasting.md)）。其后一轮收尾关闭了最后一项性能留档（图节点 parents 定长槽化：allocs −10~−22%，逐位等价），最近的工作把 LTC 的 ODE 展开融合为单个图节点——step 与 unroll 基准提速约 2.1–2.3×，对旧图路径逐位等价——并新增 `nn.UnrollRemat` 分块 BPTT：梯度逐位等价、峰值内存 O(chunk)（机制见 [doc/zh/architecture.md](doc/zh/architecture.md)，用法见 [doc/zh/cookbook.md](doc/zh/cookbook.md) 食谱 13）。剩余路线图即 [doc/zh/pitfalls.md](doc/zh/pitfalls.md) 的技术债表：余项全部为接受风险或信息级。

完整开发历程与审计轨迹见 `PROGRESS.md`。

## 致谢

LNN 是在液态神经网络（Liquid Neural Networks）研究成果之上独立重写的 Go 实现。诚挚感谢：

- **Ramin Hasani、Mathias Lechner、Alexander Amini、Daniela Rus、Radu Grosu**——液态时间常数网络（AAAI 2021）与闭式连续时间神经网络（《自然·机器智能》4, 992–1003, 2022）的作者，本库实现的方程即出自这两篇论文；
- **Mathias Lechner** 与 [`mlech26l/ncps`](https://github.com/mlech26l/ncps) 参考实现——本库 LTC 与接线语义的对照基准；
- [`raminmh/CfC`](https://github.com/raminmh/CfC) 官方 CfC 代码的作者——闭式细胞的交叉验证来源；
- **MIT CSAIL、TU Wien、IST Austria 与 Liquid AI** 团队——液态神经网络的推进与开源。

本 Go 移植版的一切错误与设计取舍均由我们自行承担。

## 开发

```
make          # gofmt -w, go vet, go test ./... -count=1 -race
make cover    # 完整测试运行，外加 coverage.txt 和按函数的覆盖率汇总
make build    # go build ./...
```

## License

MIT — 见 [LICENSE](LICENSE)。
