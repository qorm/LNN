> [English](../persistence.md) | 中文

# 持久化：Save、Load 与 LNNS 格式

**摘要：** `serialize` 包以一种紧凑、带版本、小端序（little-endian）的二进制格式（魔数（magic）`"LNNS"`）持久化张量，`nn` 在其上为 LTC、CfC 细胞与 Linear 层构建了六个 Save/Load 函数。`optimizer` 包另以 `SaveState`/`LoadState`（魔数 `"LNO1"`）持久化 SGD/Momentum/Adam 的按参数状态，使续训（resume）轨迹与不间断训练逐位一致。加载路径把输入视为**不可信字节流**（untrusted byte stream）：一切失败都是 error（绝不 panic），尺寸声明在任何缓冲区分配**之前**完成校验，恶意流的分配量只与它**实际送达**的字节数成正比。

**读者对象：** 需要为训练好的模型做检查点（checkpoint）的工程师；以及任何想逐字节弄清线上格式（wire format）及其安全契约的人。

## 两层结构

| 层 | 包 | 职责 |
|---|---|---|
| 张量流 | `github.com/qorm/LNN/serialize` | 线上格式本体：写入/读取一片 `*tensor.Tensor`，或一片参数的 `Data`。单独暴露，是为了让格式本身可以被独立审计。 |
| 模型持久化 | `github.com/qorm/LNN/nn` | `SaveLTC`/`LoadLTC`、`SaveCfC`/`LoadCfC`、`SaveLinear`/`LoadLinear`——一个 kind 字节 + 一段小头部 + 一条张量流，外加模型级校验（掩码、反转电位（reversal potential）、`unfolds`/`units`/`inDim` 上限）。 |
| 优化器状态 | `github.com/qorm/LNN/optimizer` | `SaveState`/`LoadState`——`"LNO1"` 状态流（state stream）：头部 + 存在性记录段 + 一条张量 blob，刻意复用 serialize 经审计的张量纪律（见下）。 |

## API 总览

六个模型级函数：

| 函数 | 签名 | 流内容 |
|---|---|---|
| `nn.SaveLTC(w, c)` / `nn.LoadLTC(r)` | `func(io.Writer, *LTC) error` / `func(io.Reader) (*LTC, error)` | kind `0`，头部 `inDim, units, unfolds`，17 个张量 |
| `nn.SaveCfC(w, c)` / `nn.LoadCfC(r)` | `func(io.Writer, *CfC) error` / `func(io.Reader) (*CfC, error)` | kind `1`，头部 `inDim, units`，17 个张量 |
| `nn.SaveLinear(w, l)` / `nn.LoadLinear(r)` | `func(io.Writer, *Linear) error` / `func(io.Reader) (*Linear, error)` | kind `2`，无头部，`W` 与 `B` |

两个优化器级函数（完整格式与契约见下文专节）：

| 函数 | 签名 | 流内容 |
|---|---|---|
| `optimizer.SaveState(w, o, params)` / `optimizer.LoadState(r, o, params)` | `func(io.Writer, Optimizer, []*autograd.Variable) error` / `func(io.Reader, Optimizer, []*autograd.Variable) error` | 魔数 `"LNO1"`；kind `0/1/2` = SGD/Momentum/Adam；数量；存在性记录；一条张量 blob |

`serialize` 中的四个流级构件：

| 函数 | 职责 |
|---|---|
| `serialize.WriteTensors(w, ts)` / `serialize.ReadTensors(r)` | 以线上格式写入/读取一片张量 |
| `serialize.WriteParameters(w, params)` / `serialize.LoadParameters(r, params)` | 写入 `[]*autograd.Variable` 的 `Data`；将数值**原位**（in place）读回（见下） |

### 保存与加载整个模型

```go
// 把训练好的细胞（和读出层）存进文件。
f, _ := os.Create("cfc.model")
err := nn.SaveCfC(f, cell) // 真实代码里每次调用都要检查 error
f.Close()

// 加载回来——涉及的任何 RNG 的 seed 都无关紧要。
r, _ := os.Open("cfc.model")
loaded, err := nn.LoadCfC(r)
r.Close()
// loaded.Step(x, h, ts) 与 cell.Step 的输出现在逐位相同。
```

### 训练中途做参数检查点

`serialize.LoadParameters` 将数值**原位**拷贝回给定的变量：`*autograd.Variable` 指针保持其身份（identity），因此引用它们的一切图边都依然有效——你可以在训练中途做检查点，而不必重建模型或计算图：

```go
var buf bytes.Buffer
err := serialize.WriteParameters(&buf, params)   // 快照每个参数的 p.Data
// ……之后，在同一进程或一个参数形状完全相同的新进程里……
err = serialize.LoadParameters(&buf, params)     // 数值原位恢复
for _, p := range params {
    p.ZeroGrad() // 见下文的陈旧 Grad 契约
}
```

随之而来有两条契约：

- **所有形状在任何拷贝之前完成校验。** 数量不符或任何形状不符都是 error，且失败的加载让每个参数保持原样，分毫不动。
- **陈旧的 `Grad` 被刻意保留。** 加载覆写每个参数的 `Data`，但不动它的 `Grad`：在更早的图上累积的梯度会以陈旧值（stale value）的形式存活下来。要在新图中复用这些变量的调用方，先调用 `ZeroGrad`——与任何训练步之前完全一样（[training.md](training.md)）。

## 一个完整的例子：训练 → 保存 → 加载 → 续训

下面的程序在与 `examples/cfc-sequence`（[cfc.md](cfc.md)；示例位于仓库内——`git clone https://github.com/qorm/LNN.git` 后在仓库根目录运行）相同的有界累加器任务上训练一个 `CfC`，保存细胞与读出层，把它们加载进以*不同* seed 构建的*全新*模型，验证 `Step` 输出逐位相同，然后从检查点续训——配合 optimizer 包（[training.md](training.md)）与调用方负责的梯度裁剪：

```go
package main

import (
	"bytes"
	"fmt"
	"io"
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
	inDim   = 1
	units   = 8
	seqLen  = 12
	batch   = 16
	lr      = 0.05
	maxNorm = 1.0 // 全局梯度范数裁剪
	ts      = 1.0
)

func main() {
	rng := rand.New(rand.NewSource(42))

	cell := nn.NewCfC(inDim, units, nil, rng)
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewSGD(lr)

	fmt.Println("== phase 1: train 60 iterations, then save ==")
	train(cell, readout, params, opt, rng, 60, 0)

	const modelFile = "cfc.model"
	const paramFile = "readout.params"
	saveFile(modelFile, func(w io.Writer) error { return nn.SaveCfC(w, cell) })
	saveFile(paramFile, func(w io.Writer) error { return serialize.WriteParameters(w, readout.Parameters()) })
	fi1, _ := os.Stat(modelFile)
	fi2, _ := os.Stat(paramFile)
	fmt.Printf("saved cfc.model (%d bytes) + readout.params (%d bytes)\n", fi1.Size(), fi2.Size())

	fmt.Println("== phase 2: load into fresh models (different seed) ==")
	rng2 := rand.New(rand.NewSource(123)) // seed 无关紧要：Load 覆写一切由 RNG 决定的字段
	loaded := openModel(modelFile)
	readout2 := nn.NewLinear(units, 1, rng2)
	must(serialize.LoadParameters(openStream(paramFile), readout2.Parameters()))
	fmt.Println("LoadCfC + LoadParameters: ok")

	// 加载得到的细胞与保存的细胞逐位相同。
	x := autograd.Var(tensor.Uniform(rand.New(rand.NewSource(7)), -1, 1, 4, inDim))
	out1, _ := cell.Step(x, nil, 0.5)
	out2, _ := loaded.Step(x, nil, 0.5)
	same := true
	for i := range out1.Data.Data {
		if math.Float32bits(out1.Data.Data[i]) != math.Float32bits(out2.Data.Data[i]) {
			same = false
		}
	}
	fmt.Printf("bit-identical Step output after load: %v\n", same)

	fmt.Println("== phase 3: resume training from the checkpoint ==")
	params2 := nn.ParametersOf(loaded, readout2)
	opt2 := optimizer.NewSGD(lr)
	for _, p := range params2 {
		p.ZeroGrad()
	}
	train(loaded, readout2, params2, opt2, rng, 60, 60)

	fmt.Println("== hostile streams are errors, never panics ==")
	raw, _ := os.ReadFile(modelFile)
	if _, err := nn.LoadLTC(bytes.NewReader(raw)); err != nil {
		fmt.Printf("LTC loader on a CfC stream -> %v\n", err)
	}
	if _, err := nn.LoadCfC(bytes.NewReader(raw[:len(raw)/2])); err != nil {
		fmt.Printf("truncated stream           -> %v\n", err)
	}
	bad := append([]byte(nil), raw...)
	bad[13] = 99 // 张量流的 version 字节（kind + 2 个 int32 头部 + "LNNS" 魔数之后）
	if _, err := nn.LoadCfC(bytes.NewReader(bad)); err != nil {
		fmt.Printf("unknown format version     -> %v\n", err)
	}
}

// train 跑 iters 轮有界累加器任务，打印损失（在所示轮次的更新*之前*测量），
// 带一个全局迭代偏移量以便续训输出可读。
func train(cell nn.Cell, readout *nn.Linear, params []*autograd.Variable, opt optimizer.Optimizer, rng *rand.Rand, iters, offset int) {
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

		if it%20 == 0 || it == iters-1 {
			fmt.Printf("iter %3d  loss=%.6f\n", it+offset, loss.Value())
		}
	}
}

func saveFile(path string, fn func(io.Writer) error) {
	f, err := os.Create(path)
	must(err)
	must(fn(f))
	must(f.Close())
}

func openStream(path string) *bytes.Reader {
	raw, err := os.ReadFile(path)
	must(err)
	return bytes.NewReader(raw)
}

func openModel(path string) *nn.CfC {
	c, err := nn.LoadCfC(openStream(path))
	must(err)
	return c
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "FATAL:", err)
		os.Exit(1)
	}
}
```

实际输出（Go 1.26，seed 42——确定性；每个 `loss` 在该轮更新*之前*测量）：

```
== phase 1: train 60 iterations, then save ==
iter   0  loss=0.620651
iter  20  loss=0.184146
iter  40  loss=0.158232
iter  59  loss=0.078601
saved cfc.model (1859 bytes) + readout.params (71 bytes)
== phase 2: load into fresh models (different seed) ==
LoadCfC + LoadParameters: ok
bit-identical Step output after load: true
== phase 3: resume training from the checkpoint ==
iter  60  loss=0.054492
iter  80  loss=0.045694
iter 100  loss=0.041556
iter 119  loss=0.031060
== hostile streams are errors, never panics ==
LTC loader on a CfC stream -> nn: stream holds model kind 1 (CfC), not LTC (kind 0)
truncated stream           -> serialize: tensor 6: truncated stream: claims 256 data bytes but only 176 remain: unexpected EOF
unknown format version     -> serialize: unsupported format version 99 (this build reads version 1)
```

续训恰好从训练停下的地方继续：由于 `SGD` 无状态，且产生数据的 RNG 流未被打断，续训轨迹与不间断训练**逐位一致**——`iter 100` 的损失 `0.041556` 与 `examples/cfc-sequence` 完整 250 轮运行同一迭代的打印值精确吻合。对有状态优化器，仅有模型检查点是不够的——见下一节：`SaveState`/`LoadState` 让 `Momentum` 与 `Adam` 的续训轨迹同样逐位一致。

## 优化器状态流：`SaveState`/`LoadState`（`"LNO1"` 格式）

模型检查点恢复的是参数——但 `Momentum` 与 `Adam` 的*状态*（velocity 缓冲、矩估计、更新计数、偏差校正幂）过去只存活在内存里，从纯模型检查点续训会把它们静默重置：对 Adam 而言，这等于丢弃此前每一步积累的学习率自适应（偏差校正像 `t = 0` 一样重新开始）。`optimizer.SaveState`/`LoadState` 把这份状态持久化为自定界的**状态流**（state stream，魔数 `"LNO1"`——刻意区别于 `"LNNS"`：状态流*内嵌*一条张量 blob，它自身不是 blob），并使**续训（resume）与不间断训练逐位一致**：

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
	// 参照：不间断 100 轮。
	cellA := nn.NewCfC(inDim, units, nil, rand.New(rand.NewSource(7)))
	readA := nn.NewLinear(units, 1, rand.New(rand.NewSource(7)))
	paramsA := nn.ParametersOf(cellA, readA)
	ref := train(rand.New(rand.NewSource(42)), cellA, readA, paramsA,
		optimizer.NewAdamDefault(lr), total)

	// 带检查点的运行，第一阶段：相同构造，50 轮。
	cellB := nn.NewCfC(inDim, units, nil, rand.New(rand.NewSource(7)))
	readB := nn.NewLinear(units, 1, rand.New(rand.NewSource(7)))
	paramsB := nn.ParametersOf(cellB, readB)
	optB := optimizer.NewAdamDefault(lr)
	dataB := rand.New(rand.NewSource(42))
	first := train(dataB, cellB, readB, paramsB, optB, split)

	// 检查点：模型 + 读出层参数 + 优化器状态。
	var modelBuf, paramBuf, stateBuf bytes.Buffer
	must(nn.SaveCfC(&modelBuf, cellB))
	must(serialize.WriteParameters(&paramBuf, readB.Parameters()))
	must(optimizer.SaveState(&stateBuf, optB, paramsB))
	fmt.Printf("checkpoint at step %d: model %d bytes, readout %d bytes, Adam state %d bytes\n",
		split, modelBuf.Len(), paramBuf.Len(), stateBuf.Len())

	var sgdBuf bytes.Buffer
	must(optimizer.SaveState(&sgdBuf, optimizer.NewSGD(lr), paramsB))
	fmt.Printf("SGD state stream over the same %d params: %d bytes (stateless)\n",
		len(paramsB), sgdBuf.Len())

	// 第二阶段：全新对象，状态从流恢复。
	loaded, err := nn.LoadCfC(bytes.NewReader(modelBuf.Bytes()))
	must(err)
	readC := nn.NewLinear(units, 1, rand.New(rand.NewSource(123))) // seed 无关紧要
	must(serialize.LoadParameters(bytes.NewReader(paramBuf.Bytes()), readC.Parameters()))
	paramsC := nn.ParametersOf(loaded, readC)
	optC := optimizer.NewAdamDefault(lr) // 超参由目标优化器提供
	must(optimizer.LoadState(bytes.NewReader(stateBuf.Bytes()), optC, paramsC))
	second := train(dataB, loaded, readC, paramsC, optC, total-split)

	// 比对：续训的第 50..99 步 vs 不间断运行，逐位。
	resumed := append(first, second...)
	same := len(resumed) == len(ref)
	for i := range ref {
		if same && resumed[i] != ref[i] {
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

	// 恶意流：把 Adam 流喂给 Momentum 优化器。
	err = optimizer.LoadState(bytes.NewReader(stateBuf.Bytes()),
		optimizer.NewMomentum(lr, 0.9), paramsC)
	fmt.Printf("Adam stream into Momentum -> %v\n", err)
}

// train 跑 iters 轮有界累加器任务，返回每轮更新*之前*测量的损失，
// 以 float32 位模式记录。
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

实际输出（Go 1.26，确定性）：

```
checkpoint at step 50: model 1859 bytes, readout 71 bytes, Adam state 2732 bytes
SGD state stream over the same 15 params: 19 bytes (stateless)
steps 0..99 loss bits identical to uninterrupted run: true
final parameters bit-identical: true
loss: iter 49 = 0.024681, iter 99 = 0.011424
Adam stream into Momentum -> optimizer: state stream kind 2 (Adam) does not match optimizer kind 1 (Momentum)
```

**续训契约，逐位等价。** 50 步 → 检查点 → 全新对象续训 50 步 vs 不间断 100 步：第 51–100 步**逐参数（`Float32bits`）轨迹 + 逐步 loss 逐位吻合，三优化器全过**（`TestResumeBitExact{Adam,Momentum,SGD}`；终损位模式由测试钉死；红队按指令换用不同模型族——LTC 细胞——而非实施方的模型族复验通过）。Adam 的更新计数 `t` 与偏差校正幂 `pow1 = Beta1^t`、`pow2 = Beta2^t` **逐位保存**，因此续训完整继承中断运行积累的自适应；同一状态写两次得到逐字节相同的流（`TestSaveStateDeterministic`）。

### 线上格式（`"LNO1"`）

所有整数小端序；浮点为 IEEE-754 `float32` 小端序，与 `"LNNS"` 格式相同：

| 字段 | 类型 | 说明 |
|---|---|---|
| 魔数 | `[4]byte` | 恰为 `L N O 1` |
| 版本 | `uint8` | `1`；其他值按方向分流报错（偏高 → "written by a newer version… update this build"；偏低 → "no earlier layout exists"） |
| kind | `uint8` | `0` = SGD、`1` = Momentum、`2` = Adam；加载进错误的优化器类型是精确的指名错误（见上） |
| 数量 | `uint32` | 参数记录数——每参数一条，按 `SaveState` 所收 `params` 切片的次序 |
| 记录段，重复 `count` 次（SGD 为空） | `present` `uint8` | `0` = 该参数无状态，`1` = 状态随后；仅 Adam 且 present 时：`t` `uint32`（更新计数）、`pow1`/`pow2` `float32`（`Beta1^t`/`Beta2^t`，逐位保存） |
| blob | — | 单条 `serialize` 张量流，按参数次序：每个 present 的 Momentum 参数一个 velocity 张量，每个 present 的 Adam 参数 `m` 后跟 `v`，SGD **零**个张量——自定界，拒绝尾字节 |

三优化器的记录布局：**SGD** 恒为 **19 字节**自洽空流（10 字节头部加 9 字节空张量 blob——与参数个数无关；保存是恒等操作，但流仍然写出，使每种 kind 都经同一统一格式 round-trip）；**Momentum** 每个 present 参数携带一个 velocity 张量；**Adam** 每个 present 参数携带 `m` 与 `v`，外加记录段中的 `t`/`pow1`/`pow2` 计数器。超参（`LR`、`Mu`、`Beta1`/`Beta2`/`Eps`）**刻意不入流**——它们是导出字段，由调用方在目标优化器上设置，与构造时完全一样；对 Adam，流内 `pow1`/`pow2` 必须与本优化器的 `Beta1^t`/`Beta2^t` 逐位相等，因此在不同 beta 下保存的流按损坏失败（pow 位翻转一石二鸟地兼作 β 超参错配探测器：`TestLoadStateRejectsInconsistentAdamCounters`）。

### 不可信流纪律

与模型流同一纪律：**validate-all-then-apply（先全验后应用）**——一切记录、张量与计数器在任何状态写入之前完成解析与校验，因此失败的加载让目标优化器**逐位保持原样**（velocity/m/v/t/pow 分毫不动，parameter 0 不受 parameter 1 失败的牵连；`TestLoadStateRejectsShapeMismatchWithoutSideEffects`）；**只有 error，绝不 panic**——14 类对抗流全部 error（kind 互载 6 对、未知 kind、坏魔数、version 0/99 按方向分流、count 双向不符、存在性标志 `2`、形状失配三条路径、pow 位翻转、`t` 超限在 blob 解析之前拒绝、六个截断点 → `io.ErrUnexpectedEOF`、尾字节、blob 张量数、I/O 透传；`TestLoadStateRejects*` 族，外加红队 500 个变异体 0 panic——其中 76 个加载成功者重存幂等、0 失配，66 个严格前缀截断全部拒绝）；**字节预算双门禁**——29 字节恶意流仅分配 352 B、10 字节巨 count 流仅 276 B（`TestLoadStateHostileClaimsStayWithinByteBudget`）；**`maxT = 2²⁴` 仅加载侧限额**——pow 一致性校验把 `Beta^t` 重算为 `t` 次顺序乘法，因此一条 13 字节、声称 `t = 2³²−1` 的记录不得记上四十亿次乘法的账（与下面 `maxUnfolds`/`maxUnits` 相同的加载侧不对称论证：`Adam.Step` 的运行时契约不变，因为那里的步数是调用方自己的训练史，而加载的输入是不可信流，对自己的资源预算没有表决权）。

### 索引键控与陈旧键

状态以 `params` 切片中的**下标**为键——指针无法跨进程存活，因此 `LoadState` 把第 i 条记录附给 `params[i]`：**Save 与 Load 必须给定相同的参数次序。** 以 `[B, A]` 次序加载按 `[A, B]` 保存的流，若形状恰好一致，状态会被**静默错配**——这是已披露的调用方责任而非格式缺陷：两个同形张量在格式层面本质上不可区分，与优化器既已采用的指针键控通行惯例一致（红队裁决该错配为 Informational）。加载覆写给定参数的状态——present 记录逐位替换既有缓冲，absent 记录删除该条目——并**刻意保留不在 `params` 中的变量的条目**：陈旧键（stale key）存活下来，与 `LoadParameters` 的陈旧 `Grad` 同属诚实披露契约。要得到"恰为流本身、别无其他"的状态，请构造全新优化器再加载。

## 线上格式

所有整数均为小端序（`encoding/binary.LittleEndian`）；浮点为 IEEE-754 `float32`，小端序。

### 张量流（`"LNNS"` 格式）

| 字段 | 类型 | 说明 |
|---|---|---|
| 魔数 | `[4]byte` | 恰为 `L N N S`；其他值报 "not an lnn tensor stream" |
| 版本 | `uint8` | `1`（导出常量 `serialize.Version`）；其他值被拒绝 |
| 数量 | `uint32` | 张量数；流中*恰好*编码这么多个——最后一个载荷之后的尾字节按损坏拒绝 |
| 随后重复 count 次：秩 | `uint8` | `0 ≤ rank ≤ 8` |
| 形状 | `[rank]int64` | 每个维度 `≥ 0` |
| 数据 | `[size]float32` | `size` = 形状各维之积；行主序，小端序 |

### 模型流

模型流是一个字节的 kind 标签、一段小的定长头部，以及一条张量流 blob：

| 字段 | 类型 | 说明 |
|---|---|---|
| kind | `uint8` | `0` = LTC，`1` = CfC，`2` = Linear；用错的加载函数会得到精确的错误，而不是误解析 |
| 头部 | 若干 `int32` | LTC：`inDim, units, unfolds`；CfC：`inDim, units`；Linear：无 |
| blob | — | 上述张量流（`"LNNS"`、版本、数量、数据） |

头部数值在写入端必须装得进 `int32`，在读取端必须 `≥ 1`；`unfolds` 在读取端另受上限 `1024`、`units`/`inDim` 各受上限 `2048` 约束（均见下）。

### 模型 blob 内的张量次序

固定、按模型类型手写——刻意**不**对 struct 字段做反射，因此格式可以逐行审计，且在重命名私有字段的重构中保持稳定。两个细胞承载同样 17 个张量；差别只在头部：

| 下标 | 张量 | 形状 |
|---|---|---|
| 0 | 感知掩码（sensory mask） | `[inDim, units]` |
| 1 | 循环掩码（recurrent mask） | `[units, units]` |
| 2–14 | 13 个可训练参数，按 `Parameters()` 次序：`gleak, vleak, cm, mu, sigma, w, sMu, sSigma, sW, inW, inB, outW, outB` | `[units]` / `[units, units]` / `[inDim, units]` / `[inDim]`，见 [ltc.md](ltc.md) 参数表 |
| 15 | `erev`（循环反转电位） | `[units, units]` |
| 16 | `sErev`（感知反转电位） | `[inDim, units]` |

Linear 承载两个张量：`W` `[in, out]` 与 `B` `[out]`（层的维度就在 `W` 的形状里；头部只有 kind 字节）。

## 不可信流的安全契约

本库其余部分以 panic 报告误用：它的输入来自程序自身，坏形状就是调用方的 bug。序列化是刻意的例外。加载路径消费的是来自程序*之外*的字节——文件、网络、来自其他版本的检查点——它们可能损坏、截断，乃至彻头彻尾的恶意。**读路径上的一切失败都作为 error 返回，绝不 panic**；且恶意流的分配量只与它实际送达的字节数成正比：

**固定限额，先校验后分配。** 声明的秩、维度与数量先受检查，元素计数用溢出安全乘法（`math/bits.Mul64`，与 `tensor.Size` 同一纪律）：

| 限额 | 值 | 含义 |
|---|---|---|
| `maxElems` | `2^30` 个 float32 | 单个张量载荷 ≤ 4 GiB |
| `maxCount` | `2^20` 个张量 | 每条流的张量数 |
| `maxRank` | `8` | 每个张量的轴数（本库算子聚焦 1D/2D） |
| `maxUnfolds` | `1024` | **仅加载路径**，`LoadLTC`：在 blob 解析*之前*检查，因此恶意 `unfolds` 连解析与构造的账都记不上。`NewLTC` 的运行时契约不变（仍只要求 `unfolds >= 1`）：构造器的输入来自你自己的代码，而加载的输入是不可信字节流，对自己的资源预算没有表决权 |
| `maxUnits` / `maxInDim` | 各 `2048` | **仅加载路径**，`LoadLTC`/`LoadCfC`：基于与 `maxUnfolds` 相同的理由、相同的不对称性（见下），在头部、blob 解析之前检查。阶段 9 的稀疏收缩把加载期内存从 O(units³) 变为 O(units²)，上限由 `256` 重推至此。`LoadLinear` 只承载 `W` 与 `B`，不受影响 |

声称维度宽达 `1<<62` 的流会被以 error 拒绝，而不是用一个 PB 级的 `make()` 去伺候——并以分配次数做了回归测试（`TestHostileDimDoesNotAllocate`、`TestHostileCountDoesNotAllocate`：整个恶意解码过程不超过 50 次小分配）。

**为什么 `units`/`inDim` 也要封顶。** *历史——256 上限。* 阶段 9 之前，构造器把突触归约指示矩阵（indicator matrix）实体化为稠密的 `[pre·units, units]` 矩阵（`sumIndicator`/`reversalIndicator`）：循环侧两个 `units³` 个 float32，感知侧两个 `inDim·units²` 个。因此加载期内存是 **O(units³)**，而控制它的头部只有 9–13 字节——这正是 `maxUnfolds` 已经为时间轴覆盖的那个孪生面。在旧上限处（`units = inDim = 256`），持久指示阵恰为每加载细胞 `2·(256³ + 256·256²)·4 B = 256 MiB`，外加 Load 按流内极性重新焙入指示阵时最多 `max(units³, inDim·units²)·4 B = 64 MiB` 瞬时内存——有界的最坏峰值约 320 MiB。任何封顶之前，v0.2.0 红队总扫加载一条*合法*的 `units = 512` 流（送达 5 MB）却分配了 1,560 MB（放大 311 倍）；一条 13 字节、`units = 4096` 的最小攻击流则让进程尝试 `2·4096³·4 B ≈ 550 GB` 指示阵，直到操作系统直接强杀——比 panic 更糟。

*#14 销账之后——2048 上限（重推导）。* 阶段 9 的稀疏收缩（sparse contraction，见 [ltc.md](ltc.md) 稀疏收缩一节）以 `+0` 播种、末端单位阵 MatMul 的折叠（fold）取代指示阵：**无论构造器还是加载路径，都不再实体化任何 `[units², units]` 张量**（上面那次加载期瞬时重建也一并消失——分子系数是流内 `erev` 存储的行视图）。加载期内存现在是 **O(units²)**，由流自身送达的参数矩阵主导。最坏情形，全接线，`U = units = inDim`（最大合法头部）：

    流       掩码 2 + mu/sigma/w 3 + sMu/sSigma/sW 3 + erev/sErev 2
             = 10·U² 个 float32 解析后驻留                    = 40·U² B
    细胞     同样的 10·U² 参数与掩码，加单位阵 U²
             与两份接线项表 2·U² 个 int32                     = 52·U² B
    峰值     copyFields 期间流 + 细胞同时在手                 = 92·U² B

`U = 2048` → `92·2048² B ≈` **368 MiB**——与旧制 256 上限的约 320 MiB **同一预算级，容量 8 倍**。最小攻击流（`inDim = 1`）送达约 `20·U²` 字节，峰值约**送达量的 1.5 倍**：分配与送达字节保持正比——F1 契约（"恶意流的分配量只与它实际送达的字节数成正比"）如今**由根因兑现，而非暂行封堵**。`units = 4096`（旧 jetsam PoC）仍会峰值约 1.4 GiB，因此上限保留——头部检查仍在 blob 解析之前触发——但取值 2048 而非 256；同一条流现在是一个带值的 error（`nn: LTC header has units=4096, exceeding the load limit 2048`），只花寥寥数次分配。与 `maxUnfolds` 一样，这个上限仍然刻意做成**仅加载侧**：`NewLTC`/`NewCfC` 仍接受任何 `≥ 1` 的维度，因为构造器的输入是调用方自知的分配决策——且在稀疏收缩下，`units = 1024` 全接线构造器耗费 **约 32 MB**（本机实测 `NewLTC(4, 1024, …)` 总分配 36.4 MiB、`NewCfC(4, 1024, …)` 32.4 MiB；红队复测一致），而不再是旧的约 8 GiB 悬崖——而加载的输入是不可信流，对自己的资源预算没有表决权。

**按读端能力二选一的分配策略。**

- *能报告剩余长度的读端*（`bytes.Buffer`、`bytes.Reader`、`strings.Reader` 的 `Len()` 方法）：每个载荷声明都与实际剩余字节比对，因此过大或截断的声明在其缓冲区分配**之前**就被拒绝；装得下的声明由一次全尺寸的 `make` 伺候（快路径）。
- *没有长度的读端*（`io.Pipe`、`net.Conn`、`gzip.Reader`）：事先无从证明任何事，因此载荷缓冲采用渐进分配（progressive allocation）——从小处起步（至多一个 4,096 个 float32 的块，16 KiB），只随字节到达增长。一条声称 `2^30` 个元素却在 18 字节头部后停止的流，峰值内存只有几个块（约 33 KiB——加固之前是 4 GiB 的 `make`），并以 `io.ErrUnexpectedEOF` 失败。峰值分配与实际送达的字节数保持正比；完整的流最终仍是单个切片承载全部元素。

**模型级校验，按序执行。** 加载函数在构造任何细胞之前检查：kind 字节（精确匹配——跨 kind 互载是一个指名道姓的错误）；头部维度 `≥ 1`、`unfolds` 落在 `[1, 1024]`、`units`/`inDim` 落在 `[1, 2048]`（即上面的仅加载侧上限）；张量数量（细胞恰为 17，Linear 为 2）；掩码形状与头部一致且每个掩码表项恰为 `0` 或 `1`；以及反转电位——每个 `erev`/`sErev` 表项必须**恰为 `+1` 或 `−1`**（按位比较，因此 `NaN`、`±Inf`、`0` 和 `2.5` 之类的小数全部被拒）。构造器固定这些符号，训练又把电位排除在 `Parameters()` 之外，因此承载其他值的流描述的是 `NewLTC`/`NewCfC` 永远不可能产生的细胞，一律拒绝。对已有模型的形状不符（`LoadParameters`、`copyFields`）在**任何值被拷贝之前**完成校验，因此失败的加载让目标保持原样。（头部检查消息点名当前上限：`nn: LTC header has units=4096, exceeding the load limit 2048`。）

**已模糊测试。** 变异模糊（mutation fuzz）钉住了契约：`TestMutatedTensorStreamsNeverPanic` 与 `TestMutatedModelStreamsNeverPanic`（位翻转、删除、插入、块交换）。红队对初版实现跑了 7,500 个变异体（0 panic、0 静默错乱），资源耗尽加固之后又跑了 1,200 个变异体 × 双读端类别（依然 0 panic）。

## 逐位精确性

`Load` 以一个丢弃即弃的 RNG 运行构造器来重建细胞，然后用流覆写一切由 RNG 决定的字段，因此结果**与 seed 无关**：对相同输入，加载得到的细胞产生逐位相同的 `Step` 输出与 `Parameters()` 数值（在 `Float32bits` 层面比对，含 `NaN` 与 `−0`）。数值被拷进新细胞既有的存储；流中没有任何东西别名（alias）进返回的细胞。有一处加载期细节在阶段 9 变得更好：稀疏收缩（sparse contraction）的分子系数是共享 `erev`/`sErev` Data 数组的行视图，因此 `copyFields` 执行的原位覆写被收缩**自动拾取，无需任何重建**（#14 之前的设计在此处重新实体化稠密的 `[pre·units, units]` 指示阵；见 [ltc.md](ltc.md) 稀疏收缩一节）。这也是为什么整谱符号翻转的 ±1 模式照样能加载（校验认的是 ±1 值域，而不是构造器采样出的那几个特定赋值——`TestLoadLTCAcceptsFlippedReversalPattern`、`TestLoadCfCAcceptsFlippedReversalPattern`——且加载后细胞的输出确实随该模式改变）。

写路径处理的是调用方自己拥有的内存中张量；它同样返回 error 而非 panic（nil 张量、`Shape`/`Data` 不一致、秩超过 8、数量超过上限），因此 Save 循环可以统一地报告 I/O 失败。

## 版本演进：对未来诚实

`version = 1` 是本构建唯一读取的布局。带有其他版本字节的流会以 `unsupported format version N (this build reads version 1)` 失败——未来版本会**报错而非误解析**未知布局。如果线上格式有任何变更，版本号递增，`ReadTensors` 增加对旧布局的显式支持；不存在悄无声息的尽力解码。格式版本导出为 `serialize.Version`，正是为了调用方做这类检查。

## 黄金向量：被字节钉死的冻结格式

v1 冻结是被强制执行的，而不只是被文档声明。`serialize/testdata/` 保存着提交的黄金字节流——`golden_v1_ltc.lnns`、`golden_v1_cfc.lnns`、`golden_v1_linear.lnns`（1607、1603、120 字节）——各由一个固定的、全文档化的细胞构建（`nn.NewLTC(4, 6, nil, 6, …101)`、`nn.NewCfC(4, 6, nil, …202)`、`nn.NewLinear(6, 3, …303)`）；配套的 `golden_v1_<kind>.expected.txt` 则以 `%08x` 的 float32 位模式记录加载后的细胞必须复现的精确 `Step`/`Forward` 输出，人可以逐字节审计。三个测试扮演三种不同角色，而它们执行的冻结是**按平台分级**的：

- **格式布局**——magic、version、张量计数、张量序、rank、shape 以及小端 float32 编码——在**一切平台上逐字节冻结**。写端在任何平台上对相同数值发射相同字节。
- **浮点载荷的逐位复现**是**平台与工具链内部**的保证，跨架构则不然——这是 Go 语言的行为，而非本库的缺陷：语言规范允许实现"将多个浮点运算合并为单次舍入的融合运算"（FMA 缩合），即把逐步舍入的非融合路径压成一次舍入。因此 arm64 构建与 amd64 构建之间可以合法地相差每次缩合 ≤ 1 ULP（CI 实测恰为 1 ULP：`0xbe8aa433` 对 `0xbe8aa430`），而缩合链会逐级累积——CfC 的构造参数经 Box-Muller 初始化依次执行 log/sqrt/sin/cos，CI 实测最大漂移达 6 ULP。黄金向量在 arm64（Apple Silicon）上生成，故在 `GOARCH=arm64` 上以下断言保持严格——逐位、逐字节——而在其他任何架构上，骨架仍逐字节冻结，每个载荷元素的断言窗口为 **16 ULP**（约为实测最大值的 2.7 倍；仍紧到足以拒绝真实损坏——`TestGoldenULPToleranceDiscriminates` 钉死了它的鉴别力：32 ULP 必败，形状与计数不符必败）。

- **`TestGoldenStreamsLoadBitExact`——行为冻结。** 每条提交的流都能加载，且加载后细胞的输出在生成平台上与期望位模式精确吻合（以 `Float32bits` 比对，因此 `NaN` 与 `−0` 各自与自身相等），在其他平台上落入 16 ULP 窗口。
- **`TestGoldenWriterStability`——字节级冻结。** 从文档化 seed 重建每个细胞再重新保存，在生成平台上得到的流与提交版本逐字节相同（`bytes.Equal`）；在其他平台上，封装头与线上骨架（magic、version、count、rank、shape）仍逐字节比对，仅浮点载荷按 ULP 窗口比对。
- **`TestGoldenStreamsLoadOnBothReaderClasses`——读端一致。** 已知长度快路径与渐进流式路径加载同一条黄金流，得到逐位相同的细胞——这是同一二进制内的自洽对照，因此在一切平台上保持逐位严格。

再生成是门禁式的：`TestWriteGoldenFiles` **默认跳过**，除非运行显式传入 `-write-golden`（`go test ./serialize -write-golden`），因此黄金向量只能通过一次刻意的、可见的测试运行来变更——绝不偶然发生，也绝不作为某个无关改动的副作用。CfC 黄金流反映的是阶段 8 `erev` 焙入*之后*的细胞：其加载后的 `Step` 输出与原细胞逐位相同，而这恰是那次焙入被要求保持的等价性。

---

训练循环见 [training.md](training.md)，面向用户的安全边界见 [pitfalls.md](pitfalls.md)，`serialize` 所坐落的分层见 [architecture.md](architecture.md)（它只导入 `tensor` 与 `autograd`）。
