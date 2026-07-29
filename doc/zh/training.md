> [English](../training.md) | 中文

# 用 lnn 训练模型

**摘要：** lnn 刻意不附带优化器——你自己写一个四行循环（清零梯度 → 前向 → `Backward` → 显式参数更新）作用于 `Parameters()`，稳定性（学习率与梯度裁剪（gradient clipping））也由你自己掌控。本指南端到端地展示受支持的范式。

**读者对象：** 即将用本库训练第一个模型的工程师。

## 循环

每一次训练迭代都是：

```
1. 对每个参数 ZeroGrad              // 梯度会累加；每次都要重置
2. 前向：构建一张全新的计算图       // 算子即时执行，图被记录下来
3. loss.Backward()                  // 每张图一次 Backward
4. 原地更新参数                     // 对 p.Data / p.Grad 写朴素 Go 代码
```

参数是叶 `*autograd.Variable`；更新它们意味着直接写它们的 `Data` 缓冲区。一个完整、可运行的程序（用手搓线性模型拟合 `y = 2x + 1`）：

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

	const n = 32
	xs, ys := make([]float32, n), make([]float32, n)
	for i := range xs {
		xs[i] = rng.Float32()*2 - 1
		ys[i] = 2*xs[i] + 1
	}
	x := autograd.Const(tensor.FromData(xs, n, 1)) // Const 标明"不可训练"
	y := autograd.Const(tensor.FromData(ys, n, 1))

	w := autograd.Var(tensor.Randn(rng, 1, 1))
	b := autograd.Var(tensor.New(1))

	const epochs, lr = 200, 0.1
	for epoch := 0; epoch < epochs; epoch++ {
		pred := autograd.Add(autograd.MatMul(x, w), b)
		diff := autograd.Sub(pred, y)
		loss := autograd.MeanAll(autograd.Hadamard(diff, diff))

		w.ZeroGrad()
		b.ZeroGrad()
		loss.Backward()

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

实际输出（Go 1.26，seed 42——确定性）：

```
epoch   0  loss=1.398637  w=0.5503  b=0.1674
epoch  40  loss=0.006162  w=1.8658  b=0.9795
epoch  80  loss=0.000047  w=1.9883  b=0.9982
epoch 120  loss=0.000000  w=1.9990  b=0.9998
epoch 160  loss=0.000000  w=1.9999  b=1.0000
epoch 199  loss=0.000000  w=2.0000  b=1.0000
```

三条引擎不会替你强制执行的纪律：

- **每轮迭代都构建新图**（上面的循环每轮重跑前向算子）。对*同一张*图调用两次 `Backward` 是有定义的，但会让叶梯度翻倍——见 [pitfalls.md](pitfalls.md)。
- **`ZeroGrad` 要在 `Backward` 之前，绝不在之后**——梯度按设计会跨调用、跨图地累加进叶节点。
- **前向与 `Backward` 之间不要修改参数的 `Data`：** 少数反向闭包在反向时读取父节点数据，因此原地更新必须严格放在反向传播（backpropagation）之后。

## 从 nn 模块聚合参数

`nn.Module` 就是 `interface{ Parameters() []*autograd.Variable }`，而 `nn.ParametersOf` 把若干模块展平成一个切片——这个切片*就是*你的优化器状态：

```go
cell := nn.NewLTC(1, 8, nil, 4, rng)      // 13 个可训练参数张量
readout := nn.NewLinear(8, 1, rng)        // + W 和 B
params := nn.ParametersOf(cell, readout)  // 15 个变量

for it := 0; it < iters; it++ {
	// ... 构建 xs，展开（unroll），计算 loss ...
	for _, p := range params {
		p.ZeroGrad()
	}
	loss.Backward()
	for _, p := range params {
		if p.Grad == nil { // 本图中未用到的参数，梯度为 nil
			continue
		}
		for i := range p.Data.Data {
			p.Data.Data[i] -= lr * p.Grad.Data[i]
		}
	}
}
```

`examples/ltc-sequence` 就是这个范式的完整版，任务需要跨步记忆（一个有界累加器）：`go run ./examples/ltc-sequence` 训练 250 轮迭代，损失从 `0.690761` 降到 `0.041996`。

## 为什么没有优化器

刻意为之。本库的契约是一个小而可读、可审计的数值核心；一条更新规则就是五行 Go 代码，自己写出来，学习率调度、裁剪和正则化就都留在你的代码里——可见、可 diff，而不是藏在某个框架抽象背后。图引擎对叶节点如何更新不做任何假设——SGD、动量和裁剪全都从 `p.Data`/`p.Grad` 访问中自然得出。

路线图（未实现，无时间表）：内置优化器和 CfC 细胞。它们落地之后，上面的手写范式依然有效。

## 梯度裁剪：必须做

朴素的 `float32` SGD 没有任何内置稳定化措施，而 LTC 尤其会产生大的梯度尖峰（它的 ODE 分母只由 `eps = 1e-8` 保护，而除法梯度按 `1/b²` 缩放——`1e-8` 的除数会把梯度放大约 `1e16` 倍）。一次尖峰就可能把参数推进溢出区，而 `float32` 有去无回。在玩具问题以上的一切场景中，对**全局梯度范数**做裁剪——正如 `examples/ltc-sequence` 所做：

```go
const lr, maxNorm = 0.05, 1.0

// 在 loss.Backward() 之后：缩放更新步，使全局梯度范数
// 永远不超过 maxNorm。
var norm2 float64
for _, p := range params {
	if p.Grad == nil {
		continue
	}
	for _, g := range p.Grad.Data {
		norm2 += float64(g) * float64(g) // 用 float64 累加
	}
}
scale := lr
if norm := math.Sqrt(norm2); norm > maxNorm {
	scale = lr * maxNorm / norm
}
for _, p := range params {
	if p.Grad == nil {
		continue
	}
	for i := range p.Data.Data {
		p.Data.Data[i] -= float32(scale) * p.Grad.Data[i]
	}
}
```

实测（`examples/ltc-sequence` 的首轮迭代，seed 42，units=8，unfolds=4，12 步序列）：梯度范数 `2.50` → 更新缩放 `0.019975`，而不是 `0.05`。全部 250 轮迭代中观测到的最大梯度范数为 `6.04`；裁剪在大多数早期迭代中都会触发——这正是该任务上收敛与发散的分界线。

### 可选：动量

动量就是同样的五行代码加一个速度缓冲；别无其他：

```go
vel := tensor.New(1) // 每个参数一个速度缓冲，形状相同
const beta = 0.9

// 循环内部，替换朴素 SGD 更新：
for i := range w.Data.Data {
	vel.Data[i] = beta*vel.Data[i] + w.Grad.Data[i]
	w.Data.Data[i] -= lr * vel.Data[i]
}
```

已验证：从随机初值（例如 seed 11 的 `tensor.Randn`）出发最小化 `(w - 5)²`，`lr=0.05`、`beta=0.9`，150 轮迭代后到达 `w = 5.0007`。注意动量在 `vel` 中存储的是*未缩放*的梯度；如果与范数裁剪组合使用，要么在把梯度加入速度之前套用同一个 `scale`，要么对速度本身做裁剪——二选一并保持一致。

## 我的训练为什么发散了？

按各原因出现频率排序的检查清单：

| 症状 | 可能原因 | 修复 |
|---|---|---|
| 几步之内 loss → `NaN` | lr 过大；参数过冲进入 `Exp`/`Log` 溢出区 | lr 缩小 3–10 倍；加范数裁剪 |
| 许多正常步之后突然 `NaN` | 某个 `Log(x)` 遇到了 `x ≤ 0`，或某个 `Div` 遇到了零除数（前向得 `+Inf`，反向得 `Inf` 梯度） | 钳制 `Log` 的输入；让除数远离 0 |
| LTC 损失尖峰/发散 | 来自 `1/(den+eps)` 除法的梯度尖峰（`den` 可以逼近 `eps = 1e-8`） | 全局范数裁剪（见上）；更小的 lr |
| 一步之后全部 `NaN` | `float32` 溢出扩散：一旦有一个元素是 `NaN`，非 MatMul 路径会把它带过整张图 | 裁剪；检查 `ts` 合理（`ts ≥ 1e-3` 保持完整物理保真度——见 [ltc.md](ltc.md)） |
| 梯度恰好大了一倍 | 对同一张图调用了两次 `Backward`，或漏了 `ZeroGrad` | 每张新图一次 `Backward`；先对所有参数 `ZeroGrad` |
| 梯度有限但错误 | 参数 `Data` 在前向与反向之间被修改 | 更新严格放在 `Backward` 之后 |
| 缓慢蠕动、不收敛 | lr 过小，或损失的均值化掩盖了进展 | 用根目录 README 的快速上手程序做合理性检查 |
