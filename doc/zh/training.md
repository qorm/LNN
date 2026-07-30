> [English](../training.md) | 中文

# 用 lnn 训练模型

**摘要：** 用 lnn 训练就是一个四阶段循环（清零梯度 → 前向 → `Backward` → 参数更新）作用于 `Parameters()`，稳定性（学习率与梯度裁剪（gradient clipping））由你自己掌控。更新阶段有两条路：五行人手写的 Go——理解引擎的基础；或者一次 `optimizer.Step` 调用（`optimizer` 包提供 SGD/Momentum/Adam）——推荐的生产训练形态。本指南端到端地展示这两种范式。

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

	"github.com/qorm/LNN/autograd"
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
- **前向与 `Backward` 之间不要修改参数的 `Data`：** 少数反向步骤在反向时读取父节点数据，因此原地更新必须严格放在反向传播（backpropagation）之后。

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

## 使用 optimizer 包

`optimizer` 包把循环的第 4 阶段——就是上面手写的更新，同样的 `float32` 算术、同样对 `p.Data` 的原地写入——打包成一个可审计的方法调用。三个显式 struct，没有配置对象，没有反射：

| 优化器 | 构造器 | 更新规则 |
|---|---|---|
| `SGD` | `optimizer.NewSGD(lr)` | `p -= LR·g`——就是手写循环本身 |
| `Momentum` | `optimizer.NewMomentum(lr, mu)` | `v = Mu·v + g`，然后 `p -= LR·v`——速度（velocity）存储*未缩放*的梯度，与下文"可选：动量"片段逐字一致 |
| `Adam` | `optimizer.NewAdam(lr, beta1, beta2, eps)` 或 `optimizer.NewAdamDefault(lr)`（Kingma & Ba 推荐的 0.9 / 0.999 / 1e-8） | 带偏差校正（bias correction）的一阶/二阶矩估计 |

三者都实现 `optimizer.Optimizer`（`Step(params []*autograd.Variable)`），构造器校验参数并以带违规值的消息 panic，而 `SGD` 与本指南开头快速上手中的循环逐行等价——实测：每个打印 epoch 的输出完全一致，seed 42 下 epoch 199 恢复 `w = 2.0000, b = 1.0000`。

用 `Step` 的循环：

```go
params := nn.ParametersOf(cell, readout)
opt := optimizer.NewAdamDefault(0.01)

for it := 0; it < iters; it++ {
	for _, p := range params {
		p.ZeroGrad()
	}
	loss := ...       // 构建一张全新的图
	loss.Backward()   // 每张图一次 Backward
	opt.Step(params)  // 原地更新
}
```

### Step 契约

- **`Step` 从不调用 `ZeroGrad`。** 叶梯度按设计会跨 `Backward` 调用累加，何时重置是*你的*契约。普通训练每轮迭代前清零；若想每 `N` 轮清零一次、同时在每次 `Backward` 后都 `Step`，就免费得到了跨 `N` 个微批的梯度累积（gradient accumulation）——一个替调用方清零的优化器会悄无声息地破坏这个范式。
- **`Grad` 为 nil 的参数被跳过。** 上一张图中未用到的参数（例如经 `nn.ParametersOf` 交进来的未用模块）保留其 `Data`，对有状态的优化器也保留其状态。Adam 的按参数步数计数器也不会推进，因此该参数第一次真正的更新所带的偏差校正与全新优化器完全一致。
- `Step` 假定 `p.Grad` 与 `p.Data` 形状相同——由 autograd 的 `addGrad` 保证。

### 超参是导出字段

每个超参都是普通的导出 struct 字段——直接读、直接写；训练中途调整学习率是受支持的范式：

```go
adam := optimizer.NewAdamDefault(0.1)
// ...
if epoch == 200 {
	adam.LR = 0.01 // 退火：从下一次 Step 起生效
}
```

实测（快速上手拟合，seed 42）：Adam 在 epoch 200/300 退火 `0.1 → 0.01 → 0.001`，epoch 99 时 loss 为 `0.000009`，epoch 199 到达 `w = 2.0000, b = 1.0000`。

构造器做校验；`Step` 信任字段的当前值——与手写循环信任其 `lr` 常量是同一套信任模型。因此 `optimizer.NewSGD(+Inf)` 会被*接受*（它满足 `lr > 0`；此后每步产生 `±Inf`，或在梯度恰为零的元素处产生 `NaN`），构造之后往 `adam.LR` 写入无意义的值，得到的就是你点的那份算术。

### 状态按指针键隔离

`Momentum` 与 `Adam` 以 `*autograd.Variable` 指针为键，在 map 中保存按参数的状态（速度；Adam 的矩缓冲与步数）。由此得出：

- 同一个变量反复 Step 会累积自己的状态；不同的变量永不共享状态——即使形状完全相同。
- 状态 map 会*钉住*它见过的每一个变量（map 键是强引用）：丢弃模型时连同优化器一起丢弃。
- 把变量的 `Data` 重新指向一个**同尺寸**的新张量，状态保留——优化器看到的仍是同一个参数。在两次 Step 之间**改变尺寸**则会 panic，而不是悄无声息地腐蚀更新。
- **别名变量（aliased variables）的更新相互耦合。** 建立在*同一个* `Tensor`（共享同一份 `Data` 切片）之上的两个 `Variable` 是两个不同的 map 键、却是同一个缓冲区：对两者都 Step，每一次更新都会落到共享存储上——实测 SGD `LR = 0.1`、梯度为 1 时，数值移动 `0.2` 而不是 `0.1`。把别名变量当作一个参数，只 Step 一次。

### 数值

全局 `float32`，优化器状态亦然。Adam 的更新是自归一的（`m'/sqrt(v')` 不论梯度尺度如何都保持在 `±1` 附近），因此永远不会形成宽量级累加，下文裁剪一节里全局梯度范数所用的 `float64` 技巧在这里不适用。Adam 的平方根经过 `math.Sqrt`——标准库没有 `float32` 的 sqrt，而按元素一次正确舍入的转换不是累加，不会产生漂移。

## 为什么手写循环依然保留

本库的契约是一个小而可读、可审计的数值核心；一条更新规则就是五行 Go 代码，自己写出来，学习率调度、裁剪和正则化就都留在你的代码里——可见、可 diff，而不是藏在某个框架抽象背后。图引擎对叶节点如何更新不做任何假设。`optimizer` 包（见上）把这个循环恰恰原样打包为三条常用规则，是推荐的生产训练形态；手写版本依然是理解 `Step` 在做什么的基础——也是包未覆盖的一切更新规则（权重衰减变体、奇异调度）的去处。两种形态共享同一条纪律：裁剪仍由调用方负责（下一节）。

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

`optimizer.NewMomentum(0.05, 0.9)` 就是这个片段的打包版：它以同样的算术在速度缓冲中存储未缩放的梯度，并到达同样的 `w = 5.0007`（库自带的 `TestMomentumMatchesDocExample` 把两者互为回归测试）。

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
