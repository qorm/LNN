# LNN 食谱集（cookbook）

> [English](../cookbook.md) | 中文

**目的：** 任务式食谱——"我想做 X，怎么做？"。每条食谱：一句话场景、
可直接复制的完整程序、关键行讲解、逐字实测输出，以及指向概念指南
（[training.md](training.md)、[persistence.md](persistence.md)、
[ltc.md](ltc.md)、[cfc.md](cfc.md)、[pitfalls.md](pitfalls.md)、
[shapes-and-broadcasting.md](shapes-and-broadcasting.md)、
[architecture.md](architecture.md)）中*原理*所在处的链接。

**以下每个程序都在本仓库上实测编译运行过**（Go 1.26.5，
`darwin/arm64`）；输出块为真实实测输出。想自己运行的话，把每条食谱
作为独立的 `package main`，放进一个指向本库的临时模块：

```
module lnncook

go 1.26.5

require github.com/qorm/LNN v0.0.0

replace github.com/qorm/LNN => /path/to/LNN   // 你的仓库检出路径
```

（若用已发布版本，`go get github.com/qorm/LNN@latest` 后去掉
`replace`。）所有 seed 固定，因此在 `arm64` 上你会看到与本文完全
一致的数字；在其他架构上，浮点缩合（floating-point contraction）
可能使末位略有差异——见 [faq.md](faq.md)（"不同机器上结果末位不同
正常吗？"）。

## 目录

| # | 食谱 | 一句话介绍 |
|---|---|---|
| 1 | [最小训练回路](#1-最小训练回路) | 纯 autograd 的四阶段循环：建图 → `ZeroGrad` → `Backward` → 更新。 |
| 2 | [用 optimizer 训练：Adam 加范数裁剪](#2-用-optimizer-训练adam-加范数裁剪) | `NewAdamDefault` + 调用方负责的全局梯度范数裁剪，`Step` 形态。 |
| 3 | [梯度累积](#3-梯度累积) | 每 N 次 Backward 才清零一次、Step 一次：免费获得等效大批量。 |
| 4 | [断点续训，逐位一致](#4-断点续训逐位一致) | `SaveCfC` + `SaveState`（Adam）→ 加载进全新对象 → 续训与不间断训练逐位一致。 |
| 5 | [事件驱动变 ts 序列](#5-事件驱动变-ts-序列) | `Unroll` 只接受单一 `ts`；采样间隔变化时手写 `Step` 循环。 |
| 6 | [模型检查与调试](#6-模型检查与调试) | 参数清单、按参数梯度范数、NaN/Inf 检测。 |
| 7 | [自定义损失](#7-自定义损失) | 掩码 MSE（masked MSE）与 L1：损失就是普通算子图。 |
| 8 | [LTC 与 CfC 选型](#8-ltc-与-cfc-选型) | 决策表，外加同一个训练循环驱动两种细胞。 |
| 9 | [多模块组合](#9-多模块组合) | 细胞 → Linear → Linear，`ParametersOf` 聚合，前向串联。 |
| 10 | [安全加载不可信模型文件](#10-安全加载不可信模型文件) | 错误分类范式：kind / version / 上限 / 截断。 |
| 11 | [训练中途退火学习率](#11-训练中途退火学习率) | 超参就是导出字段：写 `opt.LR`，下一次 `Step` 起生效。 |
| 12 | [确定性复现](#12-确定性复现) | 种子纪律：同样的 seed ⇒ 逐位一致的运行。 |
| 13 | [长序列训练：UnrollRemat 分块 BPTT](#13-长序列训练unrollremat-分块-bptt) | 梯度逐位一致、峰值图内存 O(chunk)：为超出全图驻留模型的序列准备的重实体化。 |

---

## 1. 最小训练回路

**场景：** 只用 `tensor` + `autograd`，以朴素梯度下降拟合
`y = 2x + 1`——库中一切都建立其上的标准四阶段循环。

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

	// 数据：y = 2x + 1，存为 [n,1] 矩阵。
	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
	}
	x := autograd.Const(tensor.FromData(xs, n, 1))
	y := autograd.Const(tensor.FromData(ys, n, 1))

	w := autograd.Var(tensor.Randn(rng, 1, 1))
	b := autograd.Var(tensor.New(1))

	const epochs, lr = 200, 0.1
	for epoch := 0; epoch < epochs; epoch++ {
		// 1. 前向：每轮迭代构建新图。
		pred := autograd.Add(autograd.MatMul(x, w), b)
		diff := autograd.Sub(pred, y)
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))

		// 2. 清零梯度，3. 反向。
		w.ZeroGrad()
		b.ZeroGrad()
		loss.Backward()

		// 4. 就地更新。
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

实测输出（seed 42——确定性）：

```
epoch   0  loss=1.398637  w=0.5503  b=0.1674
epoch  40  loss=0.006162  w=1.8658  b=0.9795
epoch  80  loss=0.000047  w=1.9883  b=0.9982
epoch 120  loss=0.000000  w=1.9990  b=0.9998
epoch 160  loss=0.000000  w=1.9999  b=1.0000
epoch 199  loss=0.000000  w=2.0000  b=1.0000
```

**关键行：**

- `autograd.Var` 标记可训练叶节点；`autograd.Const` 标注"不可训练"
  （同一函数，携带意图的别名）。
- 图每轮迭代重建——算子是即时执行（eager）并自我记录的；没有带
  （tape）对象。
- `ZeroGrad` 在 `Backward` **之前**，绝不之后：叶梯度按设计跨调用
  累加（食谱 3 正是利用这一点）。
- 更新直接写 `p.Data.Data`——参数就是你拥有的普通 `float32` 缓冲。

**延伸阅读：** [training.md](training.md) 推导了这个循环及其三条
纪律；仓库根目录 `README.md` 有同一程序作为快速上手示例；
[pitfalls.md](pitfalls.md) §3–4 讲重复 `Backward` 与前向/反向之间
改数据的陷阱。

---

## 2. 用 optimizer 训练：Adam 加范数裁剪

**场景：** 以推荐的生产范式训练一个循环模型——`optimizer.NewAdamDefault`
加调用方负责的全局梯度范数裁剪，任务是有界累加器（bounded
accumulator）序列任务（需要跨步记忆）。

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

const (
	inDim   = 1
	units   = 8
	seqLen  = 12
	batch   = 16
	iters   = 250
	maxNorm = 1.0 // 全局梯度范数裁剪
	ts      = 1.0
)

func main() {
	rng := rand.New(rand.NewSource(42))

	cell := nn.NewCfC(inDim, units, nil, rng)
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewAdamDefault(0.01)

	clips := 0
	var maxObserved float64
	var first, last float64
	for it := 0; it < iters; it++ {
		xs := make([]*autograd.Variable, seqLen)
		targets := make([]*autograd.Variable, seqLen)
		state := make([]float32, batch)
		for t := 0; t < seqLen; t++ {
			xb := make([]float32, batch)
			yb := make([]float32, batch)
			for b := 0; b < batch; b++ {
				u := float32(1)
				if rng.Intn(2) == 0 {
					u = -1
				}
				xb[b] = u
				s := state[b] + 0.25*u
				if s > 1 {
					s = 1
				} else if s < -1 {
					s = -1
				}
				state[b] = s
				yb[b] = s
			}
			xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
			targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
		}

		ys, _ := nn.Unroll(cell, xs, nil, ts)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/float32(seqLen))

		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()

		// ----- 与 doc/zh/training.md「梯度裁剪」一节逐字一致 -----
		var norm2 float64
		for _, p := range params {
			if p.Grad == nil {
				continue
			}
			for _, g := range p.Grad.Data {
				norm2 += float64(g) * float64(g)
			}
		}
		if norm := math.Sqrt(norm2); norm > maxNorm {
			s := float32(maxNorm / norm)
			for _, p := range params {
				if p.Grad == nil {
					continue
				}
				for i := range p.Grad.Data {
					p.Grad.Data[i] *= s
				}
			}
		}
		opt.Step(params)
		// ----- 逐字一致块结束 -----

		norm := math.Sqrt(norm2)
		if norm > maxNorm {
			clips++
		}
		if norm > maxObserved {
			maxObserved = norm
		}
		if it == 0 {
			first = float64(loss.Value())
		}
		last = float64(loss.Value())
		if it%50 == 0 || it == iters-1 {
			fmt.Printf("iter %3d  loss=%.6f  gnorm=%.3f\n", it, loss.Value(), norm)
		}
	}
	fmt.Printf("first=%.6f last=%.6f\n", first, last)
	fmt.Printf("clip engaged on %d/%d iterations, max observed norm %.3f\n", clips, iters, maxObserved)
}
```

实测输出（seed 42；每个 `loss` 在该轮更新*之前*测得）：

```
iter   0  loss=0.620651  gnorm=2.196
iter  50  loss=0.008872  gnorm=0.178
iter 100  loss=0.007931  gnorm=0.095
iter 150  loss=0.004200  gnorm=0.116
iter 200  loss=0.004664  gnorm=0.147
iter 249  loss=0.004286  gnorm=0.090
first=0.620651 last=0.004286
clip engaged on 14/250 iterations, max observed norm 2.196
```

**关键行：**

- 两个 `逐字一致` 标记之间的块，逐字符复制自
  [training.md](training.md) 中 `optimizer.Step` 的裁剪片段：用
  `float64` 累加全局范数，超过 `maxNorm` 时在调用 `opt.Step(params)`
  *之前*缩放**梯度**（不是缩放步长——`Step` 自己施加学习率）。
- `NewAdamDefault(lr)` 取 Kingma & Ba 的 `0.9 / 0.999 / 1e-8`；
  `Step` 跳过 `Grad` 为 nil 的参数，且从不清零梯度——那是你的契约
  （[training.md](training.md) 的 Step 契约一节）。
- Adam 的更新是自归一的，因此用 Adam 时（不同于 LTC 上的朴素
  SGD）裁剪大多在早期触发：本例 14/250 轮，首个尖峰范数
  `2.196`。裁剪仍要保留——尖峰真正来临时，它就是收敛与发散的
  分界线。

**延伸阅读：** [training.md](training.md) 的梯度裁剪一节解释为什么
在 LTC 上裁剪不可省略；仓库内的 `examples/cfc-sequence` 是同一范式
的朴素 SGD 版本，loss `0.620651 → 0.029091`。

---

## 3. 梯度累积

**场景：** 想要 `N` 个微批（micro-batch）的等效批量，但必须逐个
运行（内存所限，或数据是流式的）。叶梯度按设计就会跨 `Backward`
调用累加——因此梯度累积（gradient accumulation）根本不需要库的
任何支持：`ZeroGrad` 一次，Backward N 次，`Step` 一次。

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

// buildGraph 在给定的行上构建 y = x*w + b 的 MSE 图。
func buildGraph(x, y, w, b *autograd.Variable) *autograd.Variable {
	pred := autograd.Add(autograd.MatMul(x, w), b)
	diff := autograd.Sub(pred, y)
	return autograd.MeanAll(autograd.Hadamard(diff, diff))
}

func main() {
	rng := rand.New(rand.NewSource(42))
	const n, micro = 32, 4 // 4 个微批，每批 8 行

	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
	}

	// --- (a) 单次全批量 Backward：参考梯度 ---
	w := autograd.Var(tensor.Randn(rng, 1, 1))
	b := autograd.Var(tensor.New(1))
	xFull := autograd.Const(tensor.FromData(xs, n, 1))
	yFull := autograd.Const(tensor.FromData(ys, n, 1))
	buildGraph(xFull, yFull, w, b).Backward()
	gwFull, gbFull := w.Grad.Data[0], b.Grad.Data[0]

	// --- (b) N 个微批 Backward，开头只清零一次 ---
	// 每个微批 loss 乘以 1/micro，使累积梯度等于整个等效批量的
	// 均值梯度（每个微批的 MeanAll 只除以它自己的行数；1/micro
	// 因子把这些按批均值之和变成总均值）。
	w.ZeroGrad()
	b.ZeroGrad()
	rows := n / micro
	for m := 0; m < micro; m++ {
		xm := autograd.Const(tensor.FromData(xs[m*rows:(m+1)*rows], rows, 1))
		ym := autograd.Const(tensor.FromData(ys[m*rows:(m+1)*rows], rows, 1))
		lm := autograd.Scale(buildGraph(xm, ym, w, b), 1/float32(micro))
		lm.Backward() // 梯度累积进 w、b
	}
	fmt.Printf("full-batch    grad: dw=%+.8f db=%+.8f\n", gwFull, gbFull)
	fmt.Printf("4 micro-batch grad: dw=%+.8f db=%+.8f\n", w.Grad.Data[0], b.Grad.Data[0])
	fmt.Printf("max |diff| = %.3e (float32 addition order only)\n",
		math.Max(math.Abs(float64(w.Grad.Data[0]-gwFull)), math.Abs(float64(b.Grad.Data[0]-gbFull))))

	// --- 在累积梯度上做一次 Step ---
	opt := optimizer.NewSGD(0.1)
	wBefore := w.Data.Data[0]
	opt.Step([]*autograd.Variable{w, b})
	fmt.Printf("one Step on accumulated grads: w %.6f -> %.6f\n", wBefore, w.Data.Data[0])

	// --- 真实的累积循环：等效批量 4x32 的在线数据 ---
	rng2 := rand.New(rand.NewSource(7))
	w2 := autograd.Var(tensor.Randn(rng2, 1, 1))
	b2 := autograd.Var(tensor.New(1))
	opt2 := optimizer.NewSGD(0.1)
	const epochs, accum = 200, 4
	for e := 0; e < epochs; e++ {
		w2.ZeroGrad()
		b2.ZeroGrad() // 每 `accum` 个微批清零一次
		var lossSum float64
		for m := 0; m < accum; m++ {
			xb := make([]float32, n)
			yb := make([]float32, n)
			for i := range xb {
				xb[i] = rng2.Float32()*2 - 1
				yb[i] = 2*xb[i] + 1
			}
			l := buildGraph(autograd.Const(tensor.FromData(xb, n, 1)),
				autograd.Const(tensor.FromData(yb, n, 1)), w2, b2)
			autograd.Scale(l, 1/float32(accum)).Backward()
			lossSum += float64(l.Value())
		}
		opt2.Step([]*autograd.Variable{w2, b2}) // 每个窗口 Step 一次
		if e%40 == 0 || e == epochs-1 {
			fmt.Printf("epoch %3d  avg loss=%.6f  w=%.4f  b=%.4f\n",
				e, lossSum/accum, w2.Data.Data[0], b2.Data.Data[0])
		}
	}
}
```

实测输出（seed 42 / seed 7——确定性）：

```
full-batch    grad: dw=-0.73727763 db=-1.67409205
4 micro-batch grad: dw=-0.73727751 db=-1.67409170
max |diff| = 3.576e-07 (float32 addition order only)
one Step on accumulated grads: w 0.476583 -> 0.550311
epoch   0  avg loss=2.813679  w=0.2184  b=0.2294
epoch  40  avg loss=0.005552  w=1.8851  b=1.0031
epoch  80  avg loss=0.000021  w=1.9925  b=0.9998
epoch 120  avg loss=0.000000  w=1.9995  b=1.0000
epoch 160  avg loss=0.000000  w=2.0000  b=1.0000
epoch 199  avg loss=0.000000  w=2.0000  b=1.0000
```

**关键行：**

- `w.ZeroGrad()` 只跑**一次**，在四次 `Backward` 之前——四份按批
  梯度就求和进了同一组叶缓冲。这正是"梯度会累加"（[training.md](training.md)
  Step 契约一节："每 N 轮清零一次……就免费得到跨 N 个微批的梯度
  累积"）有文档记载的另一面。
- `autograd.Scale(l, 1/micro)` 很关键：每个微批 loss 是对*它自己*
  行数的均值，不乘 `1/micro` 的话累积梯度会是总均值梯度的
  `micro` 倍。乘上之后，二者只差 `float32` 加法顺序（上文
  `3.6e-7`）。
- `Step` 每个窗口跑一次，在所有 Backward 之后——`Step` 从不清零，
  因此消费累积梯度毫无问题。
- 如果要裁剪（食谱 2），在最后一个微批 Backward 之后、单次
  `Step` 之前，对*累积后的*梯度裁剪一次。

**延伸阅读：** [faq.md](faq.md) "为什么 Backward 后梯度还在累加"
讲累积语义；[pitfalls.md](pitfalls.md) §3 讲本食谱依赖的"重复
Backward 严格线性"保证。

---

## 4. 断点续训，逐位一致

**场景：** 训练、存检查点（模型 + 读出层 + 优化器状态），之后
加载进*全新*对象续训——并证明续训轨迹与不间断训练逐位一致。这是
[persistence.md](persistence.md) 完整示例的精简任务版。

```go
package main

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/serialize"
	"github.com/qorm/LNN/tensor"
)

const (
	inDim, units  = 1, 8
	seqLen, batch = 12, 16
	lr, ts        = 0.01, 1.0
	total, split  = 100, 50
)

func main() {
	// 参考：不间断 100 轮迭代。
	cellA := nn.NewCfC(inDim, units, nil, rand.New(rand.NewSource(7)))
	readA := nn.NewLinear(units, 1, rand.New(rand.NewSource(7)))
	paramsA := nn.ParametersOf(cellA, readA)
	ref := train(rand.New(rand.NewSource(42)), cellA, readA, paramsA,
		optimizer.NewAdamDefault(lr), total)

	// 带检查点的运行，阶段 1：相同构建，50 轮迭代。
	cellB := nn.NewCfC(inDim, units, nil, rand.New(rand.NewSource(7)))
	readB := nn.NewLinear(units, 1, rand.New(rand.NewSource(7)))
	paramsB := nn.ParametersOf(cellB, readB)
	optB := optimizer.NewAdamDefault(lr)
	dataB := rand.New(rand.NewSource(42))
	first := train(dataB, cellB, readB, paramsB, optB, split)

	// 检查点：模型 + 读出层参数 + Adam 状态。
	var modelBuf, paramBuf, stateBuf bytes.Buffer
	must(nn.SaveCfC(&modelBuf, cellB))
	must(serialize.WriteParameters(&paramBuf, readB.Parameters()))
	must(optimizer.SaveState(&stateBuf, optB, paramsB))
	fmt.Printf("checkpoint at step %d: model %d B, readout %d B, Adam state %d B\n",
		split, modelBuf.Len(), paramBuf.Len(), stateBuf.Len())

	// 阶段 2：全新对象，状态从流恢复。
	loaded, err := nn.LoadCfC(bytes.NewReader(modelBuf.Bytes()))
	must(err)
	readC := nn.NewLinear(units, 1, rand.New(rand.NewSource(123))) // seed 无关紧要
	must(serialize.LoadParameters(bytes.NewReader(paramBuf.Bytes()), readC.Parameters()))
	paramsC := nn.ParametersOf(loaded, readC)
	optC := optimizer.NewAdamDefault(lr)
	must(optimizer.LoadState(bytes.NewReader(stateBuf.Bytes()), optC, paramsC))
	second := train(dataB, loaded, readC, paramsC, optC, total-split)

	// 与不间断运行逐位比对。
	resumed := append(first, second...)
	same := true
	for i := range ref {
		if resumed[i] != ref[i] {
			same = false
		}
	}
	fmt.Printf("steps 0..%d loss bits identical to uninterrupted run: %v\n", total-1, same)
	sameP := true
	for i := range paramsA {
		a, c := paramsA[i].Data.Data, paramsC[i].Data.Data
		for j := range a {
			if math.Float32bits(a[j]) != math.Float32bits(c[j]) {
				sameP = false
			}
		}
	}
	fmt.Printf("final parameters bit-identical: %v\n", sameP)
	fmt.Printf("loss: iter %d = %.6f, iter %d = %.6f\n",
		split-1, math.Float32frombits(ref[split-1]), total-1, math.Float32frombits(ref[total-1]))
}

// train 在有界累加器任务上跑 iters 轮迭代，把每轮更新*之前*测得的
// loss 以 float32 位模式返回。
func train(rng *rand.Rand, cell nn.Cell, readout *nn.Linear,
	params []*autograd.Variable, opt optimizer.Optimizer, iters int) []uint32 {
	losses := make([]uint32, 0, iters)
	for it := 0; it < iters; it++ {
		xs := make([]*autograd.Variable, seqLen)
		targets := make([]*autograd.Variable, seqLen)
		state := make([]float32, batch)
		for t := 0; t < seqLen; t++ {
			xb := make([]float32, batch)
			yb := make([]float32, batch)
			for b := 0; b < batch; b++ {
				u := float32(1)
				if rng.Intn(2) == 0 {
					u = -1
				}
				xb[b] = u
				s := state[b] + 0.25*u
				if s > 1 {
					s = 1
				} else if s < -1 {
					s = -1
				}
				state[b] = s
				yb[b] = s
			}
			xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
			targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
		}
		ys, _ := nn.Unroll(cell, xs, nil, ts)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/float32(seqLen))
		losses = append(losses, math.Float32bits(loss.Value()))
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()
		opt.Step(params)
	}
	return losses
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "FATAL:", err)
		os.Exit(1)
	}
}
```

实测输出（确定性）：

```
checkpoint at step 50: model 1859 B, readout 71 B, Adam state 2732 B
steps 0..99 loss bits identical to uninterrupted run: true
final parameters bit-identical: true
loss: iter 49 = 0.024681, iter 99 = 0.011424
```

**关键行：**

- 三条流、三个保存器：`nn.SaveCfC`（细胞——LTC 用 `nn.SaveLTC`）、
  `serialize.WriteParameters`（读出层的 `[]*autograd.Variable`）、
  `optimizer.SaveState`（Adam 的矩、步数与偏差校正幂——`"LNO1"`
  格式）。
- 加载进**全新**对象；加载侧的任何 RNG seed 都无关紧要，因为
  `Load` 会覆写每一个源自 RNG 的字段。`LoadParameters` *就地*
  拷回数值，变量指针保持自身同一性。
- 对 Adam/Momentum 而言仅有模型检查点是不够的：没有
  `SaveState`/`LoadState`，续训会悄无声息地把偏差校正重置到
  `t = 0`。有它之后，全部 100 步逐步 loss 与不间断运行逐位吻合
  （`Float32bits`），最终参数亦然。
- **超参不在流里：** 构建目标优化器时要带上训练时用的同一组
  `LR`/beta。`LoadState` 会校验 `Beta1^t`/`Beta2^t` 与保存的幂
  逐位一致，因此 beta 不匹配会作为损坏高声失败。
- **参数顺序就是键：** `LoadState` 把第 *i* 条记录挂到
  `params[i]`——保存与加载要传同样的模块顺序（此处
  `nn.ParametersOf(loaded, readC)` 与 `nn.ParametersOf(cellB, readB)`
  一致）。

**延伸阅读：** [persistence.md](persistence.md)——完整的「训练 →
保存 → 加载 → 续训」程序（含恶意流演示）、逐字节的 `"LNO1"`
线上格式、三优化器的逐位续训契约；[faq.md](faq.md) "如何用 Adam
续训？"。

---

## 5. 事件驱动变 ts 序列

**场景：** 数据是以不规则间隔到达的事件流（传感器事件、交易、
脉冲），事件之间的时间差应当驱动细胞动力学。`nn.Unroll` 对整个
序列只接受*单一* `ts`——想要逐步变化的时间跨度，就自己写那三行
`Step` 循环。

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/tensor"
)

// unrollVariableTs 就是 nn.Unroll 内部会跑的循环，只是每步 ts 不同
// ——事件驱动 / 变步长形态。
func unrollVariableTs(cell nn.Cell, xs []*autograd.Variable, h0 *autograd.Variable, dts []float64) ([]*autograd.Variable, *autograd.Variable) {
	h := h0
	ys := make([]*autograd.Variable, len(xs))
	for i, x := range xs {
		var y *autograd.Variable
		y, h = cell.Step(x, h, dts[i]) // 每个事件一个时间跨度
		ys[i] = y
	}
	return ys, h
}

func main() {
	rng := rand.New(rand.NewSource(42))
	cell := nn.NewCfC(1, 6, nil, rng) // LTC 驱动方式完全相同：同一 Cell 接口

	// 不规则采样传感器：事件在 0.2、1.0、3.0、0.05、1.5 个时间
	// 单位的间隔后到达。距上一事件的间隔就是 ts。
	gaps := []float64{0.2, 1.0, 3.0, 0.05, 1.5}
	xs := make([]*autograd.Variable, len(gaps))
	for i := range xs {
		xs[i] = autograd.Var(tensor.Uniform(rng, -1, 1, 2, 1)) // 批量 2
	}

	// 1. 事件驱动循环：ts 跟随采样间隔。
	ys, h := unrollVariableTs(cell, xs, nil, gaps)
	fmt.Println("variable-ts unroll (ts = inter-event gap):")
	for i, y := range ys {
		fmt.Printf("  step %d  ts=%.2f  out[0]=%+.6f\n", i, gaps[i], y.Data.Data[0])
	}
	fmt.Printf("  final state finite: %v\n", allFinite(h.Data.Data))

	// 2. 同样输入、固定 ts，作对照。
	ysFix, _ := nn.Unroll(cell, xs, nil, 1.0)
	fmt.Println("fixed-ts unroll (ts = 1.0 every step):")
	for i, y := range ysFix {
		fmt.Printf("  step %d            out[0]=%+.6f\n", i, y.Data.Data[0])
	}

	// 3. 整个变 ts 序列仍是一张图：一次 Backward。
	target := autograd.Const(tensor.New(2, 1))
	var acc *autograd.Variable
	for i, y := range ys {
		d := autograd.Sub(y, target)
		sq := autograd.Hadamard(d, d)
		if i == 0 {
			acc = sq
		} else {
			acc = autograd.Add(acc, sq)
		}
	}
	loss := autograd.MeanAll(acc)
	for _, p := range cell.Parameters() {
		p.ZeroGrad()
	}
	loss.Backward()
	finite := true
	for _, p := range cell.Parameters() {
		if !allFinite(p.Grad.Data) {
			finite = false
		}
	}
	fmt.Printf("loss=%.6f  all 13 parameter grads finite through variable-ts BPTT: %v\n",
		loss.Value(), finite)

	// 4. ts 契约：必须为正且有限，否则 panic。
	for _, bad := range []float64{0, -1, math.Inf(1), math.NaN()} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("Step with ts=%v -> panic: %v\n", bad, r)
				}
			}()
			cell.Step(xs[0], nil, bad)
		}()
	}
}

func allFinite(s []float32) bool {
	for _, v := range s {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}
```

实测输出（seed 42）：

```
variable-ts unroll (ts = inter-event gap):
  step 0  ts=0.20  out[0]=+0.034594
  step 1  ts=1.00  out[0]=+0.111270
  step 2  ts=3.00  out[0]=+0.134097
  step 3  ts=0.05  out[0]=+0.135209
  step 4  ts=1.50  out[0]=+0.168719
  final state finite: true
fixed-ts unroll (ts = 1.0 every step):
  step 0            out[0]=+0.111878
  step 1            out[0]=+0.120359
  step 2            out[0]=+0.136230
  step 3            out[0]=+0.149414
  step 4            out[0]=+0.171723
loss=0.090011  all 13 parameter grads finite through variable-ts BPTT: true
Step with ts=0 -> panic: nn.CfC.Step: ts must be positive and finite, got 0
Step with ts=-1 -> panic: nn.CfC.Step: ts must be positive and finite, got -1
Step with ts=+Inf -> panic: nn.CfC.Step: ts must be positive and finite, got +Inf
Step with ts=NaN -> panic: nn.CfC.Step: ts must be positive and finite, got NaN
```

**关键行：**

- 这个循环正是 `nn.Unroll` 内部所做之事，只是把单一 `ts` 换成
  `dts[i]`——把状态 `h` 逐步传递下去，`nil` 表示零初态。
- **ts 选取：** 取 ODE 时间单位下的事件间隔。小 `ts` 几乎不推进
  膜电位（第 3 步，间隔 0.05：输出几乎不动）；大 `ts` 让它弛豫
  （relax）向稳态。把一个时间单位锚定到某个物理量（一秒、一个
  采样间隔），间隔用它表达。
- 整个变 `ts` 序列仍是一张图：一次 `Backward` 就对所有步、所有
  时间跨度求导。
- `ts` 必须为正且有限——`0`、负数、`±Inf`、`NaN` 都会 panic
  （如上演示）。

**延伸阅读：** [ltc.md](ltc.md) 的 ts 一节——完整 `ts` 契约与有限
性域（`ts ≳ 1e-3` 保持完整物理保真度；`≈ 1e-38` 以下仅为有限性
域）；[cfc.md](cfc.md) 中 CfC 的同一契约；[faq.md](faq.md) "ts 怎么
选？"。

---

## 6. 模型检查与调试

**场景：** 在信任一个模型之前——或者当一次运行行为异常时——枚举
它的参数、检查数值中有没有非有限元素，并在一次 Backward 之后查看
按参数的梯度范数。

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/tensor"
)

// hasNaNInf 扫描缓冲中的非有限值。
func hasNaNInf(s []float32) bool {
	for _, v := range s {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return true
		}
	}
	return false
}

// gradNorm 返回梯度缓冲的 L2 范数，以 float64 累加。
func gradNorm(g []float32) float64 {
	var n2 float64
	for _, v := range g {
		n2 += float64(v) * float64(v)
	}
	return math.Sqrt(n2)
}

func main() {
	rng := rand.New(rand.NewSource(42))
	cell := nn.NewLTC(2, 6, nil, 4, rng)
	readout := nn.NewLinear(6, 1, rng)
	params := nn.ParametersOf(cell, readout)

	// 参数清单：下标、形状、元素数。LTC 的参数顺序记录在
	// doc/zh/ltc.md 的参数表中。
	names := []string{
		"gleak", "vleak", "cm", "mu", "sigma", "w", "sMu", "sSigma", "sW",
		"inW", "inB", "outW", "outB", "readout.W", "readout.B",
	}
	total := 0
	fmt.Println("parameter inventory:")
	for i, p := range params {
		fmt.Printf("  [%2d] %-10s shape %-7s %4d elems\n", i, names[i], fmt.Sprint(p.Data.Shape), len(p.Data.Data))
		total += len(p.Data.Data)
		if hasNaNInf(p.Data.Data) {
			fmt.Printf("       !! non-finite values at init\n")
		}
	}
	fmt.Printf("  total %d params, %d trainable elements\n", len(params), total)

	// 一次前向/反向，然后做按参数诊断。
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 3, 2))
	y := autograd.Var(tensor.Uniform(rng, -1, 1, 3, 1))
	for _, p := range params {
		p.ZeroGrad()
	}
	ys, _ := nn.Unroll(cell, []*autograd.Variable{x, x, x, x}, nil, 1.0)
	diff := autograd.Sub(readout.Forward(ys[3]), y)
	loss := autograd.MeanAll(autograd.Hadamard(diff, diff))
	loss.Backward()

	fmt.Printf("loss=%.6f (finite: %v)\n", loss.Value(), !hasNaNInf([]float32{loss.Value()}))
	fmt.Println("per-parameter gradient diagnostics:")
	var global float64
	for i, p := range params {
		switch {
		case p.Grad == nil:
			fmt.Printf("  [%2d] %-10s grad=nil  (parameter unused in this graph)\n", i, names[i])
		case hasNaNInf(p.Grad.Data):
			fmt.Printf("  [%2d] %-10s GRAD HAS NaN/Inf — clip/inspect before stepping\n", i, names[i])
		default:
			n := gradNorm(p.Grad.Data)
			global += n * n
			fmt.Printf("  [%2d] %-10s |grad|=%.6f\n", i, names[i], n)
		}
	}
	fmt.Printf("global gradient norm: %.6f\n", math.Sqrt(global))
}
```

实测输出（seed 42）：

```
parameter inventory:
  [ 0] gleak      shape [6]        6 elems
  [ 1] vleak      shape [6]        6 elems
  [ 2] cm         shape [6]        6 elems
  [ 3] mu         shape [6 6]     36 elems
  [ 4] sigma      shape [6 6]     36 elems
  [ 5] w          shape [6 6]     36 elems
  [ 6] sMu        shape [2 6]     12 elems
  [ 7] sSigma     shape [2 6]     12 elems
  [ 8] sW         shape [2 6]     12 elems
  [ 9] inW        shape [2]        2 elems
  [10] inB        shape [2]        2 elems
  [11] outW       shape [6]        6 elems
  [12] outB       shape [6]        6 elems
  [13] readout.W  shape [6 1]      6 elems
  [14] readout.B  shape [1]        1 elems
  total 15 params, 185 trainable elements
loss=0.546932 (finite: true)
per-parameter gradient diagnostics:
  [ 0] gleak      |grad|=0.100045
  [ 1] vleak      |grad|=0.703703
  [ 2] cm         |grad|=0.027202
  [ 3] mu         |grad|=1.205924
  [ 4] sigma      |grad|=0.051544
  [ 5] w          |grad|=0.215859
  [ 6] sMu        |grad|=0.226093
  [ 7] sSigma     |grad|=0.014812
  [ 8] sW         |grad|=0.108447
  [ 9] inW        |grad|=0.168447
  [10] inB        |grad|=0.218458
  [11] outW       |grad|=0.157862
  [12] outB       |grad|=1.150598
  [13] readout.W  |grad|=0.676386
  [14] readout.B  |grad|=0.989655
global gradient norm: 2.221342
```

**关键行：**

- `ParametersOf` 返回的就是普通 `*autograd.Variable`：
  `p.Data.Shape` 与 `p.Data.Data` 给出形状与数值；下标→名称的
  映射就是文档记载的参数顺序（[ltc.md](ltc.md) 参数表——细胞
  `Parameters()` 的顺序被持久化格式冻结）。
- `p.Grad == nil` 表示该参数未参与上一张图——对未使用的模块是
  预期行为，对已接线的模块则是 bug。
- 此处的全局范数（`2.22`）超过典型的 `maxNorm = 1.0` 裁剪阈值
  ——正是食谱 2 的裁剪会缩放掉的部分。
- `hasNaNInf` 是一行式健康检查：可疑一步之后扫 `p.Data.Data`，
  Step 之前扫 `p.Grad.Data`。

**loss 不降？按出现频率排序的检查清单：**

1. 学习率过大（loss → `NaN`）或过小（缓慢爬行）——[training.md](training.md)
   末尾的症状表。
2. 漏了 `ZeroGrad`，或对同一张图做了两次 `Backward`——梯度恰好
   翻倍（[pitfalls.md](pitfalls.md) §3）。
3. LTC 上没有梯度裁剪——`1/(den+eps)` 除法可把梯度尖峰放大
   ~`1e16` 倍（[training.md](training.md) 梯度裁剪一节）。
4. 在前向与 `Backward` 之间改了参数 `Data`——梯度有限但是错的
   （[pitfalls.md](pitfalls.md) §4）。
5. 形状 bug——loss 沿错误的轴取平均，或广播悄无声息地做了你没
   想要的事（[shapes-and-broadcasting.md](shapes-and-broadcasting.md)）。

**延伸阅读：** [faq.md](faq.md) "出现 NaN 损失" 与 "为什么 Backward
后梯度还在累加"。

---

## 7. 自定义损失

**场景：** 任务需要非标准损失——带缺失条目的标签（掩码 MSE，
masked MSE），或你希望对其稳健的离群值（L1/MAE）。没有要学的
损失 API：损失就是一张以标量结尾的普通算子图，
[shapes-and-broadcasting.md](shapes-and-broadcasting.md) 算子表中
的一切都能用。

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/tensor"
)

// fit 跑 300 轮朴素 SGD；lossFn 从 (pred, y) 构建损失。
func fit(x, y *autograd.Variable, w0 float32, lossFn func(pred, y *autograd.Variable) *autograd.Variable) (float32, float32, float32) {
	w := autograd.Var(tensor.FromData([]float32{w0}, 1, 1))
	b := autograd.Var(tensor.New(1))
	const lr = 0.05
	var loss *autograd.Variable
	for epoch := 0; epoch < 300; epoch++ {
		pred := autograd.Add(autograd.MatMul(x, w), b)
		loss = lossFn(pred, y)
		w.ZeroGrad()
		b.ZeroGrad()
		loss.Backward()
		w.Data.Data[0] -= lr * w.Grad.Data[0]
		b.Data.Data[0] -= lr * b.Grad.Data[0]
	}
	return loss.Value(), w.Data.Data[0], b.Data.Data[0]
}

func main() {
	rng := rand.New(rand.NewSource(42))
	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	mask := make([]float32, n)
	valid := 0
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
		mask[i] = 1
		valid++
		if i%4 == 0 { // 每第 4 个标签损坏：+50 离群值
			ys[i] += 50
			mask[i] = 0
			valid--
		}
	}
	x := autograd.Const(tensor.FromData(xs, n, 1))
	y := autograd.Const(tensor.FromData(ys, n, 1))
	m := autograd.Const(tensor.FromData(mask, n, 1)) // 掩码：图常量
	w0 := tensor.Randn(rng, 1, 1).Data[0]            // 三次拟合同一起点
	fmt.Printf("data: y = 2x + 1, %d/%d labels corrupted by +50 outliers; w0 = %.4f\n", n-valid, n, w0)

	// (1) 掩码 MSE——掩码就是图中一个普通张量：把损坏行的平方
	// 误差清零，再重缩放为有效行上的均值。
	loss, w, b := fit(x, y, w0, func(pred, y *autograd.Variable) *autograd.Variable {
		diff := autograd.Sub(pred, y)
		sq := autograd.Hadamard(diff, diff)
		return autograd.Scale(autograd.MeanAll(autograd.Hadamard(m, sq)), float32(n)/float32(valid))
	})
	fmt.Printf("masked MSE : final loss=%.6f  w=%.4f  b=%.4f   <- recovers (2, 1)\n", loss, w, b)

	// (2) 对照：对所有行的朴素 MSE。
	loss, w, b = fit(x, y, w0, func(pred, y *autograd.Variable) *autograd.Variable {
		diff := autograd.Sub(pred, y)
		return autograd.MeanAll(autograd.Hadamard(diff, diff))
	})
	fmt.Printf("plain MSE  : final loss=%.6f  w=%.4f  b=%.4f   <- outliers pull the fit\n", loss, w, b)

	// (3) L1 / MAE：|pred - y| 也只是另一张算子图。
	loss, w, b = fit(x, y, w0, func(pred, y *autograd.Variable) *autograd.Variable {
		return autograd.MeanAll(autograd.Abs(autograd.Sub(pred, y)))
	})
	fmt.Printf("L1 / MAE   : final loss=%.6f  w=%.4f  b=%.4f   <- robust to outliers\n", loss, w, b)
}
```

实测输出（seed 42；真实关系为 `y = 2x + 1`）：

```
data: y = 2x + 1, 8/32 labels corrupted by +50 outliers; w0 = 0.4766
masked MSE : final loss=0.000000  w=1.9999  b=1.0000   <- recovers (2, 1)
plain MSE  : final loss=461.407043  w=-2.9410  b=12.9715   <- outliers pull the fit
L1 / MAE   : final loss=12.506966  w=1.9891  b=0.9969   <- robust to outliers
```

**关键行：**

- 掩码是一个 `Const` 张量；`Hadamard(m, sq)` 把损坏行的平方误差
  清零，`Scale(..., n/valid)` 把全行均值变成有效行上的均值。被
  掩行恰好得到零梯度——任何参数都看不到那 `+50` 离群值。
- L1 就是 `MeanAll(Abs(diff))`——`Abs` 的反向是 `sign`（恰为 0 处
  取次梯度 0），不论离群值多大梯度都有界：L1 拟合落在
  `(1.9891, 0.9969)`，而朴素 MSE 被拖到 `(−2.9410, 12.9715)`。
- 训练循环什么都不用改：`lossFn` 返回什么标量图，`.Backward()`
  就对什么求导。

**延伸阅读：** [shapes-and-broadcasting.md](shapes-and-broadcasting.md)
有完整的算子/形状表；若你的损失用到 `Log`/`Div`，见
[pitfalls.md](pitfalls.md) §2 中无保护的域名（钳制 `Log` 输入、
让除数远离 0）。

---

## 8. LTC 与 CfC 选型

**场景：** 该用哪种细胞？二者以不同积分器离散化*同一个*膜 ODE
——取舍在于图成本、精度与便利性，而非表达能力。

| | LTC（`nn.NewLTC`） | CfC（`nn.NewCfC`） |
|---|---|---|
| 积分器 | 半隐式欧拉（semi-implicit Euler），`unfolds` 个子步 | Lemma 1 闭式解（closed-form solution），单步解析 |
| 每 RNN 步的图节点数 | 自阶段 16 起为单个融合节点（子步不再录图；内核暂存仍随 `unfolds` 增长） | 自阶段 18 起为单个融合节点——任意维度 24 个节点 |
| 反向内存 | `∝ units × unfolds × seqLen` 保留到 `Backward`（融合内核的暂存） | `∝ units × seqLen`——无 `unfolds` 因子 |
| 大 `ts` 行为 | 弛豫向稳态（隐式格式，稳定） | 弛豫向稳态（对冻结激活精确） |
| 变 `ts` | 支持——逐步（[食谱 5](#5-事件驱动变-ts-序列)） | 支持——同一契约 |
| 相对 ODE 的精度 | 对 `ts/unfolds` 一阶；增大 `unfolds` 可收紧 | 对 `ts` 一阶（Lemma 1 冻结激活）；`ts → 0` 时两者收敛 |
| 构造器 | `NewLTC(inDim, units, wiring, unfolds, rng)` | `NewCfC(inDim, units, wiring, rng)`——无 `unfolds` |

经验法则：内存或长序列吃紧时**从 CfC 起步**（根本没有 `unfolds`
因子）；想复现 ncps 数字、或要参考实现的精确欧拉动力学时**选
LTC**。其余一切——13 个可训练张量及其初始化区间、固定的 ±1
反转电位（reversal potential）、接线（wiring）、`ts` 契约、
`Cell` 接口——都是共享的。

由于参数抽取顺序相同，两种细胞可以在 `nn.Cell` 背后直接互换
——同一个循环可以训练任何一种：

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

const (
	inDim, units  = 1, 8
	seqLen, batch = 12, 16
	lr, ts        = 0.05, 1.0
	iters         = 250
)

func main() {
	// 1. 同一 seed -> 两种细胞的初始化逐位一致（参数抽取顺序相同）。
	ltc := nn.NewLTC(inDim, units, nil, 4, rand.New(rand.NewSource(42)))
	cfc := nn.NewCfC(inDim, units, nil, rand.New(rand.NewSource(42)))
	pl, pc := ltc.Parameters(), cfc.Parameters()
	same := len(pl) == len(pc)
	for i := range pl {
		a, b := pl[i].Data.Data, pc[i].Data.Data
		for j := range a {
			if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
				same = false
			}
		}
	}
	fmt.Printf("%d trainable tensors each; same-seed init bit-identical: %v\n", len(pl), same)

	// 2. 两种细胞满足同一组接口：一个通用 train 函数驱动任何
	// 一种——换构造器，循环保留。
	fmt.Printf("LTC accumulator task: final loss %.6f\n", train(ltc, rand.New(rand.NewSource(7))))
	fmt.Printf("CfC accumulator task: final loss %.6f\n", train(cfc, rand.New(rand.NewSource(7))))
}

// trainableCell 同时被 *nn.LTC 与 *nn.CfC 满足：Cell 接口
// （Step/StateSize）加 Module（Parameters）。
type trainableCell interface {
	nn.Cell
	nn.Module
}

// train 与细胞类型无关。
func train(cell trainableCell, rng *rand.Rand) float32 {
	readout := nn.NewLinear(units, 1, rand.New(rand.NewSource(99)))
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewSGD(lr)
	var last float32
	for it := 0; it < iters; it++ {
		xs := make([]*autograd.Variable, seqLen)
		targets := make([]*autograd.Variable, seqLen)
		state := make([]float32, batch)
		for t := 0; t < seqLen; t++ {
			xb := make([]float32, batch)
			yb := make([]float32, batch)
			for b := 0; b < batch; b++ {
				u := float32(1)
				if rng.Intn(2) == 0 {
					u = -1
				}
				xb[b] = u
				s := state[b] + 0.25*u
				if s > 1 {
					s = 1
				} else if s < -1 {
					s = -1
				}
				state[b] = s
				yb[b] = s
			}
			xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
			targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
		}
		ys, _ := nn.Unroll(cell, xs, nil, ts)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/float32(seqLen))
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()
		opt.Step(params)
		last = loss.Value()
	}
	return last
}
```

实测输出（seed 42 / 7 / 99——确定性）：

```
13 trainable tensors each; same-seed init bit-identical: true
LTC accumulator task: final loss 0.033956
CfC accumulator task: final loss 0.044693
```

**关键行：**

- `nn.Cell`（Step/StateSize）两种细胞都满足；把它与 `nn.Module`
  （Parameters）组合，就得到一个通用训练循环可以接受的接口
  ——把 `NewLTC(..., unfolds, rng)` 换成 `NewCfC(..., rng)`，
  其余一概不变。
- 同一 seed ⇒ 两种细胞初始化逐位一致，因此 A/B 对比从同一点
  出发；上文末损的微小差异来自两种积分器，而非初始化。
- 内存：全接线细胞的加载期峰值为 `92·U²` 字节，`U = units =
  inDim`（[persistence.md](persistence.md)）；*训练*图把每步成本
  乘以 `seqLen`，并且——仅对 LTC——再乘以 `unfolds`
  （[pitfalls.md](pitfalls.md) §9）。

**延伸阅读：** [ltc.md](ltc.md) 与 [cfc.md](cfc.md)——逐式对照，
以及实测的 LTC→CfC 收敛（单步 `~O(ts²)`）；[faq.md](faq.md) "LTC
和 CfC 该用哪个？"。

---

## 9. 多模块组合

**场景：** 真实模型是若干模块串联——细胞 → 隐藏层 → 读出层。
`ParametersOf` 把它们全部摊平为一个切片，你的循环与优化器就作用
于其上；前向串联就是普通的函数复合。

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

func main() {
	rng := rand.New(rand.NewSource(42))

	// 组合：CfC 细胞 -> 隐藏 Linear -> 读出 Linear。
	cell := nn.NewCfC(2, 8, nil, rng) // Step 输出：[batch, 8]
	hidden := nn.NewLinear(8, 16, rng)
	readout := nn.NewLinear(16, 1, rng)

	// ParametersOf 把三个模块摊平为一个切片——它就是优化器的
	// 全世界。顺序：细胞的 13 个张量，然后隐藏层 W/B，再读出层 W/B。
	params := nn.ParametersOf(cell, hidden, readout)
	fmt.Printf("modules: CfC(2->8) + Linear(8->16) + Linear(16->1)\n")
	fmt.Printf("parameters: %d tensors (%d cell + 2 + 2)\n", len(params), len(cell.Parameters()))

	// 前向串联：每个 Forward 都是一张普通算子图；自由复合。
	x := autograd.Var(tensor.Uniform(rng, -1, 1, 4, 2)) // batch 4
	y, h := cell.Step(x, nil, 1.0)                      // y: [4, 8], h: [4, 8]
	z := hidden.Forward(y)                              // z: [4, 16]
	pred := readout.Forward(z)                          // pred: [4, 1]
	fmt.Printf("shapes: x %v -> cell %v (state %v) -> hidden %v -> readout %v\n",
		x.Data.Shape, y.Data.Shape, h.Data.Shape, z.Data.Shape, pred.Data.Shape)

	// 在玩具目标上训练整条链。
	opt := optimizer.NewAdamDefault(0.01)
	target := autograd.Const(tensor.FromData([]float32{0.5, -0.5, 0.25, -0.25}, 4, 1))
	for it := 0; it < 300; it++ {
		y, _ := cell.Step(x, nil, 1.0)
		pred := readout.Forward(hidden.Forward(y))
		diff := autograd.Sub(pred, target)
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()
		opt.Step(params)
		if it == 0 || it == 299 {
			fmt.Printf("iter %3d  loss=%.6f\n", it, loss.Value())
		}
	}
}
```

实测输出（seed 42）：

```
modules: CfC(2->8) + Linear(8->16) + Linear(16->1)
parameters: 17 tensors (13 cell + 2 + 2)
shapes: x [4 2] -> cell [4 8] (state [4 8]) -> hidden [4 16] -> readout [4 1]
iter   0  loss=0.167217
iter 299  loss=0.000001
```

**关键行：**

- `nn.ParametersOf(cell, hidden, readout)`——对 `nn.Module`
  （`interface{ Parameters() []*autograd.Variable }`）可变参；
  返回的 17 张量切片就是你 `ZeroGrad`、交给 `Step`、（配合
  `serialize.WriteParameters`）存检查点的对象。**保存/加载之间
  保持其顺序稳定**（[食谱 4](#4-断点续训逐位一致)）。
- 前向即复合：`readout.Forward(hidden.Forward(y))`。每次
  `Forward`/`Step` 调用都向当前图添加节点，因此一次 `Backward`
  同时穿过读出层、隐藏层与全部细胞动力学求导。
- 形状流 `[batch, inDim] → [batch, units] → [batch, 16] → [batch, 1]`；
  `Linear.Forward` 接受 `[batch, in]`，返回 `[batch, out]`。

**延伸阅读：** [training.md](training.md) 参数聚合一节讲
`ParametersOf` 与未使用模块的 nil-`Grad` 规则；
[architecture.md](architecture.md) 讲图节点是什么。

---

## 10. 安全加载不可信模型文件

**场景：** 一个模型文件从进程之外到来——下载、上传、别组的
检查点。把它当恶意输入对待：加载路径上的一切失败都是 `error`
（绝不 panic），而消息会告诉你失败属于*哪一类*。分类，然后
处置。

```go
package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"strings"

	"github.com/qorm/LNN/nn"
)

// classify 把加载路径错误映射到面向操作者的桶。此处一切失败都是
// error，绝不 panic——这是持久化契约（doc/zh/persistence.md）。
func classify(err error) string {
	switch {
	case err == nil:
		return "ok"
	case errors.Is(err, io.ErrUnexpectedEOF):
		return "truncated stream (interrupted write / bad copy) — safe to retry"
	case strings.Contains(err.Error(), "model kind"):
		return "wrong loader for this file (kind mismatch) — use the matching LoadXxx"
	case strings.Contains(err.Error(), "unsupported format version"):
		return "format version skew — newer writer or corruption; update the library"
	case strings.Contains(err.Error(), "bad magic"):
		return "not an LNN stream at all — wrong file"
	case strings.Contains(err.Error(), "load limit"):
		return "model exceeds the load-path caps — hostile or oversized header"
	default:
		return "other validation failure — inspect the message"
	}
}

func attempt(name string, f func() error) {
	err := f()
	fmt.Printf("%-28s -> %-28s (%v)\n", name, classify(err), err)
}

func main() {
	// 一条合法的 CfC 流，用来交叉加载与破坏。
	var buf bytes.Buffer
	cell := nn.NewCfC(2, 4, nil, rand.New(rand.NewSource(1)))
	if err := nn.SaveCfC(&buf, cell); err != nil {
		panic(err)
	}
	raw := buf.Bytes()
	fmt.Printf("legit CfC stream: %d bytes\n\n", len(raw))

	attempt("garbage bytes", func() error {
		_, err := nn.LoadCfC(bytes.NewReader([]byte("this is not a model file..")))
		return err
	})
	attempt("empty file", func() error {
		_, err := nn.LoadCfC(bytes.NewReader(nil))
		return err
	})
	attempt("truncated at half", func() error {
		_, err := nn.LoadCfC(bytes.NewReader(raw[:len(raw)/2]))
		return err
	})
	attempt("LTC loader on CfC file", func() error {
		_, err := nn.LoadLTC(bytes.NewReader(raw))
		return err
	})
	attempt("Linear loader on CfC file", func() error {
		_, err := nn.LoadLinear(bytes.NewReader(raw))
		return err
	})

	badVer := append([]byte(nil), raw...)
	badVer[13] = 99 // 内嵌张量流的 version 字节
	attempt("corrupt version byte", func() error {
		_, err := nn.LoadCfC(bytes.NewReader(badVer))
		return err
	})

	// 构造一个声明 units = 4096 的 LTC 头：kind + 3 个 int32 + blob。
	// 头上限检查在 blob 解析之前触发。
	huge := make([]byte, 1+12+9)
	huge[0] = 0                                   // kind LTC
	binary.LittleEndian.PutUint32(huge[1:], 4)    // inDim
	binary.LittleEndian.PutUint32(huge[5:], 4096) // units
	binary.LittleEndian.PutUint32(huge[9:], 4)    // unfolds
	copy(huge[13:], []byte("LNNS"))
	huge[17] = 1 // version
	attempt("units=4096 header", func() error {
		_, err := nn.LoadLTC(bytes.NewReader(huge))
		return err
	})
	hugeU := make([]byte, 1+12+9)
	hugeU[0] = 0
	binary.LittleEndian.PutUint32(hugeU[1:], 4)
	binary.LittleEndian.PutUint32(hugeU[5:], 8)
	binary.LittleEndian.PutUint32(hugeU[9:], 4096) // unfolds
	copy(hugeU[13:], []byte("LNNS"))
	hugeU[17] = 1
	attempt("unfolds=4096 header", func() error {
		_, err := nn.LoadLTC(bytes.NewReader(hugeU))
		return err
	})

	// 正常路径仍然可用。
	attempt("legit CfC stream", func() error {
		_, err := nn.LoadCfC(bytes.NewReader(raw))
		return err
	})
}
```

实测输出：

```
legit CfC stream: 827 bytes

garbage bytes                -> wrong loader for this file (kind mismatch) — use the matching LoadXxx (nn: stream holds model kind 116 (unknown), not CfC (kind 1))
empty file                   -> truncated stream (interrupted write / bad copy) — safe to retry (nn: reading model kind: unexpected EOF)
truncated at half            -> truncated stream (interrupted write / bad copy) — safe to retry (serialize: tensor 7: truncated stream: claims 64 data bytes but only 11 remain: unexpected EOF)
LTC loader on CfC file       -> wrong loader for this file (kind mismatch) — use the matching LoadXxx (nn: stream holds model kind 1 (CfC), not LTC (kind 0))
Linear loader on CfC file    -> wrong loader for this file (kind mismatch) — use the matching LoadXxx (nn: stream holds model kind 1 (CfC), not Linear (kind 2))
corrupt version byte         -> format version skew — newer writer or corruption; update the library (serialize: unsupported format version 99 (this build reads version 1): the stream was written by a newer version of the library; update this build to read it)
units=4096 header            -> model exceeds the load-path caps — hostile or oversized header (nn: LTC header has units=4096, exceeding the load limit 2048)
unfolds=4096 header          -> model exceeds the load-path caps — hostile or oversized header (nn: LTC header has unfolds=4096, exceeding the load limit 1024)
legit CfC stream             -> ok                           (<nil>)
```

**关键行：**

- 截断会包装 `io.ErrUnexpectedEOF`——用 `errors.Is` 匹配，别用
  字符串匹配。其余按消息前缀分类（`nn:` 模型级、`serialize:`
  张量流级、`optimizer:` 状态流级）。
- 交叉加载是*具名*错误，不是误解析：加载器先读一字节的 kind
  标签，并告诉你文件实际装的是什么（"stream holds model kind 1
  (CfC)"）。
- 头上限检查在张量 blob 解析**之前**：一条 22 字节、声明
  `units = 4096` 的流只花几次分配，而不是 ~1.4 GiB。加载路径
  限额：`units`/`inDim ≤ 2048`、`unfolds ≤ 1024`（构造器不限
  ——那是你自己的分配决策）。
- 失败的加载**零副作用**：对目标模型，所有形状在任何数值拷贝
  之前完成校验，你已有的模型保持原样。版本偏差宁可拒绝也不
  猜测（`version 1` 是本构建读取的唯一布局；更高版本说"update
  this build"，更低说"corrupt or forged"）。

**延伸阅读：** [persistence.md](persistence.md) 的不可信流安全契约
一节（限额表、两类读取器、fuzz 证据）与 [pitfalls.md](pitfalls.md)
§10 的用户向摘要；[faq.md](faq.md) "加载报错怎么读？"。

---

## 11. 训练中途退火学习率

**场景：** 先快后细——训练中改变学习率。每个超参都是普通的导出
结构体字段；写它，下一次 `Step` 就用新值。没有调度器对象。

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

func main() {
	rng := rand.New(rand.NewSource(42))

	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
	}
	x := autograd.Const(tensor.FromData(xs, n, 1))
	y := autograd.Const(tensor.FromData(ys, n, 1))

	w := autograd.Var(tensor.Randn(rng, 1, 1))
	b := autograd.Var(tensor.New(1))
	params := []*autograd.Variable{w, b}

	adam := optimizer.NewAdamDefault(0.1) // 超参就是导出字段
	const epochs = 400
	for epoch := 0; epoch < epochs; epoch++ {
		pred := autograd.Add(autograd.MatMul(x, w), b)
		diff := autograd.Sub(pred, y)
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()
		adam.Step(params)

		// 退火：写字段从下一次 Step 起生效。
		if epoch == 200 {
			adam.LR = 0.01
		}
		if epoch == 300 {
			adam.LR = 0.001
		}

		switch epoch {
		case 0, 99, 199, 201, 299, 301, 399:
			fmt.Printf("epoch %3d  LR=%.3f  loss=%.6f  w=%.6f  b=%.6f\n",
				epoch, adam.LR, loss.Value(), w.Data.Data[0], b.Data.Data[0])
		}
	}
}
```

实测输出（seed 42）：

```
epoch   0  LR=0.100  loss=1.398637  w=0.576583  b=0.100000
epoch  99  LR=0.100  loss=0.000009  w=1.995972  b=0.999955
epoch 199  LR=0.100  loss=0.000000  w=1.999978  b=0.999999
epoch 201  LR=0.010  loss=0.000000  w=1.999991  b=0.999996
epoch 299  LR=0.010  loss=0.000000  w=2.000002  b=1.000001
epoch 301  LR=0.001  loss=0.000000  w=2.000002  b=1.000001
epoch 399  LR=0.001  loss=0.000000  w=2.000002  b=1.000001
```

**关键行：**

- `adam.LR = 0.01` 就是整个调度——字段 `LR`、`Beta1`/`Beta2`/`Eps`
  （Adam）、`LR`/`Mu`（Momentum）、`LR`（SGD）都是导出的，每次
  `Step` 都重新读取。
- 手写规则同样如此：`lr` 常量想怎么调度就怎么调度——循环是你
  的。
- 信任模型：构造器校验（`NewAdam` 对 `LR ≤ 0` panic），但
  `Step` 信任字段所写之值——往 `LR` 里写 `NaN`，得到的就是你
  点名的那份算术。

**延伸阅读：** [training.md](training.md) 超参一节（退火范式及其
实测结果，本食谱复现之）；食谱 4——超参刻意*不在* `"LNO1"`
状态流里，因此续训时要在目标优化器上把它们设好。

---

## 12. 确定性复现

**场景：** 让一次运行可复现——为调试、为论文、为回归测试。规则
是严格的：同样的 seed ⇒ 逐位一致的轨迹，在 `Float32bits` 层面
比较。

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

func run(seed int64) (uint32, uint32) {
	rng := rand.New(rand.NewSource(seed))
	cell := nn.NewCfC(1, 6, nil, rng)
	readout := nn.NewLinear(6, 1, rng)
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewAdamDefault(0.01)
	data := rand.New(rand.NewSource(seed + 1))
	var last uint32
	for it := 0; it < 120; it++ {
		xs := make([]*autograd.Variable, 10)
		targets := make([]*autograd.Variable, 10)
		state := make([]float32, 8)
		for t := 0; t < 10; t++ {
			xb := make([]float32, 8)
			yb := make([]float32, 8)
			for b := 0; b < 8; b++ {
				u := float32(1)
				if data.Intn(2) == 0 {
					u = -1
				}
				xb[b] = u
				s := state[b] + 0.25*u
				if s > 1 {
					s = 1
				} else if s < -1 {
					s = -1
				}
				state[b] = s
				yb[b] = s
			}
			xs[t] = autograd.Var(tensor.FromData(xb, 8, 1))
			targets[t] = autograd.Var(tensor.FromData(yb, 8, 1))
		}
		ys, _ := nn.Unroll(cell, xs, nil, 1.0)
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		loss := autograd.Scale(autograd.MeanAll(acc), 1/10.0)
		for _, p := range params {
			p.ZeroGrad()
		}
		loss.Backward()
		opt.Step(params)
		last = math.Float32bits(loss.Value())
	}
	return last, math.Float32bits(params[0].Data.Data[0])
}

func main() {
	l1, p1 := run(42)
	l2, p2 := run(42) // 再次用同样的 seed
	l3, _ := run(43)  // 不同 seed

	fmt.Printf("run A (seed 42): final loss bits %08x, param[0] bits %08x\n", l1, p1)
	fmt.Printf("run B (seed 42): final loss bits %08x, param[0] bits %08x\n", l2, p2)
	fmt.Printf("same seed, two runs: bit-identical: %v\n", l1 == l2 && p1 == p2)
	fmt.Printf("run C (seed 43): final loss bits %08x — differs: %v\n", l3, l3 != l1)
}
```

实测输出：

```
run A (seed 42): final loss bits 3c5dd2c0, param[0] bits 3f1dd476
run B (seed 42): final loss bits 3c5dd2c0, param[0] bits 3f1dd476
same seed, two runs: bit-identical: true
run C (seed 43): final loss bits 3c7f5455 — differs: true
```

**关键行：**

- 种子纪律意味着运行中*每个* `rand.Rand` 都有 seed——模型初始化
  与数据生成分开播种（如上），改一个流才不会扰动另一个。
- 比较位模式（`math.Float32bits`，含 `NaN`/`−0`），不要比较打印
  出来的十进制：两次运行可以打印出相同的 `%.6f` 却在最后一位
  不同。
- 同一 seed ⇒ 同样的 RNG 流 ⇒ 同样的图 ⇒ 同样的 `float32`
  算术——在给定平台与工具链上（[pitfalls.md](pitfalls.md) §7）。

**延伸阅读：** [persistence.md](persistence.md) 黄金向量（golden
vector）一节——跨平台的细微之处：格式布局在任何平台都逐字节
冻结，但跨架构时浮点载荷每次融合乘加（FMA）可差 ≤ 1 ULP
（arm64 对 amd64），因此黄金向量测试在生成架构之外断言 16 ULP
窗口；[faq.md](faq.md) "不同机器上结果末位不同正常吗？"。

---

## 13. 长序列训练：UnrollRemat 分块 BPTT

**场景：** 序列长到「整图驻留至 `Backward`」成为内存之墙——T = 512 时
全展开要钉住约 11.5 MB 的存活图，而 `UnrollRemat` 只保留约 0.65 MB
（约 18×；`BenchmarkUnrollPeakMemory512` /
`BenchmarkUnrollRematPeakMemory512`，chunk 16、units 8、batch 16 实测）。
`nn.UnrollRemat` 对 `lossFn(ys)` 做贯穿时间的求导，梯度与
`Unroll` + `loss.Backward()` **逐位一致**，峰值图内存为
O(chunkSize) 而非 O(len(xs))。下面的程序先对两种细胞证明逐位一致，
再用它真实训练。

```go
package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/qorm/LNN/autograd"
	"github.com/qorm/LNN/nn"
	"github.com/qorm/LNN/optimizer"
	"github.com/qorm/LNN/tensor"
)

const (
	inDim, units  = 1, 8
	seqLen, batch = 48, 16
	chunk         = 8
	lr, ts        = 0.01, 1.0
	iters         = 250
)

// makeBatch 抽取一批全新的带界累加器数据：
// s_t = clip(s_{t-1} + 0.25*u_t, -1, 1)，u_t ∈ {-1, +1}。
func makeBatch(rng *rand.Rand) (xs, targets []*autograd.Variable) {
	xs = make([]*autograd.Variable, seqLen)
	targets = make([]*autograd.Variable, seqLen)
	state := make([]float32, batch)
	for t := 0; t < seqLen; t++ {
		xb := make([]float32, batch)
		yb := make([]float32, batch)
		for b := 0; b < batch; b++ {
			u := float32(1)
			if rng.Intn(2) == 0 {
				u = -1
			}
			xb[b] = u
			s := state[b] + 0.25*u
			if s > 1 {
				s = 1
			} else if s < -1 {
				s = -1
			}
			state[b] = s
			yb[b] = s
		}
		xs[t] = autograd.Var(tensor.FromData(xb, batch, inDim))
		targets[t] = autograd.Var(tensor.FromData(yb, batch, 1))
	}
	return xs, targets
}

// mseLoss 是逐步 MSE 读出损失。它的拼写使损失图的 DFS 按升序
// （t = 0, 1, 2, …）访问各步输出——remat 的快路径；降序拼写是
// 已文档化的最坏情形（见本食谱的取舍表）。
func mseLoss(readout *nn.Linear, targets []*autograd.Variable) func(ys []*autograd.Variable) *autograd.Variable {
	return func(ys []*autograd.Variable) *autograd.Variable {
		var acc *autograd.Variable
		for t, y := range ys {
			diff := autograd.Sub(readout.Forward(y), targets[t])
			sq := autograd.Hadamard(diff, diff)
			if t == 0 {
				acc = sq
			} else {
				acc = autograd.Add(acc, sq)
			}
		}
		return autograd.Scale(autograd.MeanAll(acc), 1/float32(len(ys)))
	}
}

// bitIdentical 在同一细胞、同一批数据、同一损失上各跑一次
// 全图反向（参照）与一次 UnrollRemat，报告损失值与每个参数梯度
// 是否逐位一致。
func bitIdentical(cell interface {
	nn.Cell
	nn.Module
}, rng, data *rand.Rand) bool {
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout)
	xs, targets := makeBatch(data)
	lossFn := mseLoss(readout, targets)

	for _, p := range params {
		p.ZeroGrad()
	}
	ys, _ := nn.Unroll(cell, xs, nil, ts)
	ref := lossFn(ys)
	ref.Backward()
	refGrads := make([][]float32, len(params))
	for i, p := range params {
		refGrads[i] = append([]float32(nil), p.Grad.Data...)
	}

	for _, p := range params {
		p.ZeroGrad()
	}
	_, _, rmLoss := nn.UnrollRemat(cell, params, xs, nil, ts, chunk, lossFn)
	if math.Float32bits(ref.Value()) != math.Float32bits(rmLoss.Value()) {
		return false
	}
	for i, p := range params {
		for j := range p.Grad.Data {
			if math.Float32bits(p.Grad.Data[j]) != math.Float32bits(refGrads[i][j]) {
				return false
			}
		}
	}
	return true
}

func main() {
	// 1. 对全图反向的逐位一致性，两种细胞各验一次
	//    （LTC 会额外走过它脊柱类所需的 σ 扫描）。
	ltc := nn.NewLTC(inDim, units, nil, 4, rand.New(rand.NewSource(42)))
	cfc := nn.NewCfC(inDim, units, nil, rand.New(rand.NewSource(42)))
	fmt.Printf("T=%d chunk=%d, remat vs whole-graph backward, bit-identical: LTC %v, CfC %v\n",
		seqLen, chunk,
		bitIdentical(ltc, rand.New(rand.NewSource(1)), rand.New(rand.NewSource(7))),
		bitIdentical(cfc, rand.New(rand.NewSource(1)), rand.New(rand.NewSource(7))))

	// 2. 用 UnrollRemat 的真实训练循环：四阶段纪律不变，
	//    只是 loss 返回时反向已经完成。
	rng := rand.New(rand.NewSource(42))
	cell := nn.NewCfC(inDim, units, nil, rng)
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout) // 每一个可训练叶——有审计
	opt := optimizer.NewAdamDefault(lr)
	data := rand.New(rand.NewSource(7))
	var first, last float64
	for it := 0; it < iters; it++ {
		xs, targets := makeBatch(data)
		for _, p := range params {
			p.ZeroGrad()
		}
		_, _, loss := nn.UnrollRemat(cell, params, xs, nil, ts, chunk, mseLoss(readout, targets))
		opt.Step(params)
		if it == 0 {
			first = float64(loss.Value())
		}
		last = float64(loss.Value())
		if it%50 == 0 || it == iters-1 {
			fmt.Printf("iter %3d  loss=%.6f\n", it, loss.Value())
		}
	}
	fmt.Printf("first=%.6f last=%.6f\n", first, last)
}
```

实测输出（seed 42 / 1 / 7——确定性）：

```
T=48 chunk=8, remat vs whole-graph backward, bit-identical: LTC true, CfC true
iter   0  loss=0.750716
iter  50  loss=0.075355
iter 100  loss=0.016325
iter 150  loss=0.012297
iter 200  loss=0.015780
iter 249  loss=0.010747
first=0.750716 last=0.010747
```

**关键行：**

- `params := nn.ParametersOf(cell, readout)` 必须列出细胞 `Step`
  消费的**每一个**可训练叶：结构探针会审计完备性，缺漏时按编号
  指名 panic——漏列的叶会在每次扫描中静默多累加一份梯度
  （错成 2–3 倍）。只被损失消费的叶（读出层的）可列可不列；
  `xs`/`h0` 的梯度始终参与。
- `lossFn` 收到的是 **detach 后**的逐步输出（值逐位一致、背后无
  图），且恰好被调用一次；它的返回值在 `UnrollRemat` 返回时已经
  反向完毕——梯度如同 `loss.Backward()` 之后一样落在叶节点里，
  因此 `ZeroGrad` → `Step` 纪律照旧。返回的 `ys`/`hN` 同样是
  detach 的：可安全读取、可喂给后续计算，但无法再对它们求导
  （求导已经发生过了）。
- 「逐位一致」用 `math.Float32bits` 断言，而不是打印小数——
  两次运行可以打出相同的 `%.6f` 而末位不同（食谱 12）。T = 48
  配 chunk = 8 恰好切成 6 个重算单元。
- LTC 的检验额外走过了它的脊柱类 `cm` 所需的 σ 扫描；CfC 只走
  rest 扫描（两次前向、一次反向——remat 的理想代价）。两者都与
  全图反向逐位一致。

**chunkSize 怎么选：** 峰值图内存与 `chunkSize` 近似线性（chunk 8
约为上面 chunk 16 数字的一半），而 O(T) 的 detach 输出/状态都很小
（每步一个 `[batch, units]` 张量），因此更小的 chunk 只多付逐单元
固定开销——本引擎上 4–16 是合理区间。一个结构性告诫：被播种的步
若*不是*损失访问序的纪录高（record high），会与后继步粘连，使重算
单元合并超出 `chunkSize`。请把损失拼成按升序访问各步输出（上面的
自然累加次序），单元便恰好等于 `chunkSize`；降序访问配小 chunk 是
极端最坏情形——全部合并为一个 O(T) 单元，此时 `UnrollRemat` 的峰值
内存*和*算力都严格贵于一次全图反向（这是逐位保真的代价，不是
bug）。闭合引用参数的正则项是合法的（损失只调用一次），但必须写成
数据在前——`Add(data, penalty)`，绝不 `Add(penalty, data)`——或者
干脆做成常量；违规会被探针 panic。

| | `Unroll` + `loss.Backward()` | `UnrollRemat` |
|---|---|---|
| 峰值图内存 | O(T × 逐步子图)——T = 512：驻留约 11.5 MB | O(chunkSize × 逐步子图) + O(T) 小张量——T = 512、chunk 16：保留约 0.65 MB |
| 每次迭代算力 | 1 次前向 + 1 次反向 | CfC：2 前向 + 1 反向；LTC：3 前向 + 2 反向（σ 扫描）；非升序损失对两种细胞都再加一趟仿射补扫 |
| 梯度 | 参照基准 | 与参照逐位一致（两种细胞，见上） |
| 损失形状 | 任意 | 升序访问是快路径；对抗性访问序迫使单元合并（最坏可超全展开） |
| `params` 参数 | — | 必须列出 `Step` 消费的每一个可训练叶；有审计，缺漏 panic |
| 对细胞的要求 | 任意 `nn.Cell` | 逐步图结构必须是 `(x, h)` 的纯函数——两种内置细胞均满足；值依赖的分支会不可探地漂约 1–2 ULP（[pitfalls.md](pitfalls.md)） |
| 何时使用 | 默认；中短序列 | 长序列，或内存吃紧的部署 |

**延伸阅读：** 两遍法机制与它重放的三个折叠类见
[architecture.md](architecture.md)；留档的残余角落（最坏情形、
值依赖细胞、双 NaN 载荷角）见 [pitfalls.md](pitfalls.md)；完整
契约见 `nn.UnrollRemat` 的 godoc；一行式条目见 [api.md](api.md)。
