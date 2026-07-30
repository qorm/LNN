> [English](README.md) | 中文

# lnn

一个纯 Go 数值计算库：稠密 `float32` 张量（tensor）、反向模式自动微分，以及液态时间常数（Liquid Time-Constant，LTC）神经细胞——**零第三方依赖**（唯一的导入就是标准库）。LTC 实现遵循 Hasani 等人的论文
[Liquid Time-constant Networks](https://ojs.aaai.org/index.php/AAAI/article/view/17017)
（AAAI 2021）及参考实现 [`mlech26l/ncps`](https://github.com/mlech26l/ncps)。

lnn 小而显式。它宁可牺牲覆盖面，也要保证内核可读、可审计：没有代码生成，没有 GPU 后端，没有运算符重载技巧——只有 Go。

## 包结构

| 包 | 职责 |
|---|---|
| `lnn/tensor` | 稠密行主序 `float32` 张量，聚焦 1D/2D 的算子集：矩阵乘、带有限广播（broadcasting）的逐元素运算、激活、归约、切片、随机初始化。 |
| `lnn/autograd` | 动态计算图（computation graph）引擎。每个算子在其输出 `Variable` 上记录一个反向闭包；`Backward` 按逆拓扑序遍历计算图，将梯度累加（gradient accumulation）到叶节点（leaf）。 |
| `lnn/nn` | 神经网络构件：`Linear` 层、`Wiring` 突触（synapse）拓扑、`LTC` 液态细胞及其闭式（closed-form）兄弟细胞 `CfC`，以及在序列上驱动循环细胞的 `Cell`/`Unroll` 抽象。 |
| `lnn/optimizer` | 作用于 `autograd` 的显式参数更新规则：SGD、经典重球动量（momentum）Momentum、Adam（Kingma & Ba，含偏差校正（bias correction））。一次 `Step(params)` 调用替换手写更新循环。 |

## 文档

面向库使用者的指南位于 [`doc/`](doc/)，中文版位于 [`doc/zh/`](doc/zh/)：

| 指南 | 内容 |
|---|---|
| [doc/zh/training.md](doc/zh/training.md) | 手写训练循环与 `optimizer` 包（SGD/Momentum/Adam）、梯度裁剪（gradient clipping）、发散排查清单 |
| [doc/zh/shapes-and-broadcasting.md](doc/zh/shapes-and-broadcasting.md) | 广播规则表、归约输出形状、非对称约定 |
| [doc/zh/ltc.md](doc/zh/ltc.md) | LTC 论文↔代码对照、参数表、`ts` 契约、接线（wiring） |
| [doc/zh/cfc.md](doc/zh/cfc.md) | CfC 闭式细胞：Lemma 1 论文对照、exprel 稳定化、与 LTC 的关系 |
| [doc/zh/architecture.md](doc/zh/architecture.md) | 三层设计、计算图机制、`float32` 约束 |
| [doc/zh/pitfalls.md](doc/zh/pitfalls.md) | 并发契约、溢出场景、残余风险、路线图 |

建议的阅读顺序见 [doc/zh/README.md](doc/zh/README.md)。
各包的 API 参考以 godoc 为准：`go doc lnn/tensor`、`go doc lnn/autograd`、`go doc lnn/nn`。
按 Go 社区惯例，godoc 注释（包括三个 `doc.go`）保持英文；本中文版文档与英文 `doc/` 一一对应，可交叉查阅。

## 安装

模块路径就是裸名 `lnn`，没有 vanity import URL，因此 `go get` 无法通过网络解析它。在模块正式发布之前，请用 `replace` 指令就地引用：

```
git clone <this repository> LNN
```

```go
// your app's go.mod
module myapp

go 1.26

require lnn v0.0.0

replace lnn => ../LNN
```

在仓库内部，`make build` / `make test` 开箱即用。

## 快速上手

训练循环就是对着 `autograd` 手写的——这是理解本库的基础——而 `optimizer` 包把同一个循环打包成 SGD/Momentum/Adam 供生产使用。下面的程序用一个手搓的线性模型配合朴素 SGD 拟合 `y = 2x + 1`，只用到 `tensor` 和 `autograd`，`go run` 即可运行。把手写更新替换为 `optimizer.NewSGD(lr)` + `Step(params)`，输出完全一致——见 [doc/zh/training.md](doc/zh/training.md)。

朴素的 `float32` SGD 没有任何内置稳定化措施：请使用温和的学习率；在稍大的问题上考虑对全局梯度范数做裁剪（`examples/ltc-sequence` 将最大范数裁剪到 1.0）。

```go
package main

import (
	"fmt"
	"math/rand"

	"lnn/autograd"
	"lnn/tensor"
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

// CfC 是闭式兄弟细胞：同一条 ODE、同一套 13 参数突触参数化，
// 但没有 unfolds——Lemma 1 闭式解一步推进整个时间跨度 ts（doc/zh/cfc.md）。
cfc := nn.NewCfC(4, 8, nil, rng)
out2, h2 := cfc.Step(x, nil, 0.1)
```

`examples/ltc-sequence` 把这些拼成了一个完整的训练循环（对 `nn.ParametersOf(cell, readout)` 做手写 SGD），任务是一个玩具序列任务——运行方式：`go run ./examples/ltc-sequence`。

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
- **计算图保留到 `Backward` 为止。** 每个中间张量都被计算图持有，因此内存随算子数量增长。一次 LTC step 会把 `unfolds` 轮 ODE 迭代展开进图，自突触向量化起每轮是 O(units) 个向量块加两次 MatMul 收缩——从 O(units²) 个逐突触节点降下来（`LTCStep` 3,440 allocs/op，−53%；`UnrollBackward` 68,688，−43%）。在这个引擎上，`units`、`unfolds` 和序列长度请保持适度；`CfC` 细胞（[doc/zh/cfc.md](doc/zh/cfc.md)）则完全没有 `unfolds` 因子。

## 并发契约

**lnn 在设计上是单线程的。**

- `Backward` 会修改叶变量的 `Grad` 缓冲区；在共享参数的变量上并发运行它属于数据竞争（data race），会丢失或损坏梯度（已在 `go test -race` 下实证）。
- `Variable` 和 `Tensor` 直接暴露其存储（`Data`），且不带任何同步机制。不要从多个 goroutine 读写同一个张量。
- 用于初始化的 `math/rand.Rand` 同样不是 goroutine 安全的。

并行负载的受支持模式：给每个 goroutine 它自己的张量、变量和 RNG，绝不跨 goroutine 共享计算图或其参数。

## 状态与路线图

截至本提交的诚实成熟度评估（覆盖率为 `go test -cover` 实测）：

| 包 | 状态 |
|---|---|
| `tensor` | 核心稳定、测试充分（约 90% 行覆盖率）。部分防御性检查（溢出安全的尺寸计算、空输入边界情形）仍在加固中。 |
| `autograd` | 稳定、测试充分（约 98% 行覆盖率）；已覆盖路径上的梯度均通过有限差分检验。 |
| `nn` | 可用、测试充分（约 99% 行覆盖率）：LTC 与 CfC 的前向/反向路径有回归测试，包括闭式退化情形检验、微小/NaN `ts` 防护和接线校验。两个细胞的反转电位都是固定的 ±1 常量，不可训练。CfC 是阶段 6 的新特性，API 仍可能演进。 |
| `optimizer` | 稳定，100% 行覆盖率：三条更新规则均与独立参考实现对照验证（SGD 逐位一致，Adam 对 float64 参考最大偏差约 1.6e-6），指针键状态语义有回归测试。 |

CfC（Closed-form Continuous-time）细胞与内置优化器已在阶段 6 落地：`nn.CfC`（[doc/zh/cfc.md](doc/zh/cfc.md)）是 API 仍可能演进的新特性，`optimizer` 包（SGD/Momentum/Adam）已稳定——手写循环依然有效，也仍是理解引擎的基础。序列展开由通用的 `nn.Unroll` 助手覆盖，`examples/ltc-sequence` 展示了端到端训练范式。剩余路线图：序列化（Save/Load）——完整的技术债表追踪于 [doc/zh/pitfalls.md](doc/zh/pitfalls.md)。

修复计划与进展追踪见 `PLAN.md` 和 `PROGRESS.md`。

## 致谢

lnn 是在液态神经网络（Liquid Neural Networks）研究成果之上独立重写的 Go 实现。诚挚感谢：

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
