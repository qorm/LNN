> [English](../training.md) | 中文

# 用 LNN 训练模型

**摘要：** 用 LNN 训练就是一个四阶段循环（清零梯度 → 前向 → `Backward` → 参数更新）作用于 `Parameters()`，稳定性（学习率与梯度裁剪（gradient clipping））由你自己掌控。更新阶段有两条路：五行人手写的 Go——理解引擎的基础；或者一次 `optimizer.Step` 调用（`optimizer` 包提供 SGD/Momentum/Adam/AdEMAMix/Schedule-Free AdamW）——推荐的生产训练形态。本指南端到端地展示这两种范式。

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

`examples/ltc-sequence` 就是这个范式的完整版，任务需要跨步记忆（一个有界累加器）：`go run ./examples/ltc-sequence` 训练 250 轮迭代，损失从 `0.690761` 降到 `0.041996`。示例都在仓库内——克隆仓库（`git clone https://github.com/qorm/LNN.git`）后在仓库根目录运行。

## 使用 optimizer 包

`optimizer` 包把循环的第 4 阶段——就是上面手写的更新，同样的 `float32` 算术、同样对 `p.Data` 的原地写入——打包成一个可审计的方法调用。五个显式 struct，没有配置对象，没有反射：

| 优化器 | 构造器 | 更新规则 |
|---|---|---|
| `SGD` | `optimizer.NewSGD(lr)` | `p -= LR·g`——就是手写循环本身 |
| `Momentum` | `optimizer.NewMomentum(lr, mu)` | `v = Mu·v + g`，然后 `p -= LR·v`——速度（velocity）存储*未缩放*的梯度，与下文"可选：动量"片段逐字一致 |
| `Adam` | `optimizer.NewAdam(lr, beta1, beta2, eps)` 或 `optimizer.NewAdamDefault(lr)`（Kingma & Ba 推荐的 0.9 / 0.999 / 1e-8） | 带偏差校正（bias correction）的一阶/二阶矩估计 |
| `AdEMAMix` | `optimizer.NewAdEMAMix(lr, b1, b2, b3, alpha, warmup, eps)` 或 `optimizer.NewAdEMAMixDefault(lr, warmup)`（0.9 / 0.999 / 0.9999 / α=5 / 1e-8） | Adam 再加一条*慢速*梯度 EMA，以系数 α 混入——非常老的梯度仍保有一票 |
| `ScheduleFreeAdamW` | `optimizer.NewScheduleFreeAdamW(lr, b1, b2, eps)` 或 `optimizer.NewScheduleFreeAdamWDefault(lr)` | 不要学习率调度：在 `y` 处求梯度、在 `z` 上跑基础 AdamW、可部署权重是加权平均的 `x`——附 `Train`/`Eval` 转换契约 |

五者都实现 `optimizer.Optimizer`（`Step(params []*autograd.Variable)`），构造器校验参数并以带违规值的消息 panic，而 `SGD` 与本指南开头快速上手中的循环逐行等价——实测：每个打印 epoch 的输出完全一致，seed 42 下 epoch 199 恢复 `w = 2.0000, b = 1.0000`。

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

`Momentum`、`Adam`、`AdEMAMix` 与 `ScheduleFreeAdamW` 以 `*autograd.Variable` 指针为键，在 map 中保存按参数的状态（速度；矩缓冲、慢速 EMA 缓冲、`z` 序列与更新计数）。由此得出：

- 同一个变量反复 Step 会累积自己的状态；不同的变量永不共享状态——即使形状完全相同。
- 状态 map 会*钉住*它见过的每一个变量（map 键是强引用）：丢弃模型时连同优化器一起丢弃。
- 把变量的 `Data` 重新指向一个**同尺寸**的新张量，状态保留——优化器看到的仍是同一个参数。在两次 Step 之间**改变尺寸**则会 panic，而不是悄无声息地腐蚀更新。
- **别名变量（aliased variables）的更新相互耦合。** 建立在*同一个* `Tensor`（共享同一份 `Data` 切片）之上的两个 `Variable` 是两个不同的 map 键、却是同一个缓冲区：对两者都 Step，每一次更新都会落到共享存储上——实测 SGD `LR = 0.1`、梯度为 1 时，数值移动 `0.2` 而不是 `0.1`。把别名变量当作一个参数，只 Step 一次。

### 数值

全局 `float32`，优化器状态亦然。Adam 的更新是自归一的（`m'/sqrt(v')` 不论梯度尺度如何都保持在 `±1` 附近），因此永远不会形成宽量级累加，下文裁剪一节里全局梯度范数所用的 `float64` 技巧在这里不适用。Adam 的平方根经过 `math.Sqrt`——标准库没有 `float32` 的 sqrt，而按元素一次正确舍入的转换不是累加，不会产生漂移。

## AdEMAMix 与 Schedule-Free AdamW

两条较新的更新规则已加入本包，各自对照其论文的第三方 `float64` 参考实现移植（最大偏差分别为 9.5e-6 / 3.3e-7，带鉴别力守门确保门禁能逮住注入的变异），由 77 个顶层测试外加加载路径 fuzz 目标覆盖，并以与最初三条相同的 50+50 对 100 续训逐位契约钉住（[persistence.md](persistence.md)）。

### AdEMAMix：带长记忆的 Adam

Pagliardini、Ablin 与 Grangier，*The AdEMAMix Optimizer: Better, Faster, Older*（[arXiv:2409.03137](https://arxiv.org/abs/2409.03137)，ICLR 2025）：在 Adam 之上增加第二条**慢速**梯度 EMA，以系数 α 混入，使非常老的梯度保有不可忽略的影响力，而快速 EMA 保持灵敏。逐元素、在更新计数 `t` 处：

```
m1  = β1·m1 + (1−β1)·g          快速 EMA
m2  = β3(t)·m2 + (1−β3(t))·g    慢速 EMA——刻意不做偏差校正
v   = β2·v + (1−β2)·g²
p  −= LR · (m1/(1−β1ᵗ) + α(t)·m2) / (sqrt(v/(1−β2ᵗ)) + eps)
```

- 慢速 EMA 不校正是论文的设计而非疏漏：β3 = 0.9999 时校正因子 `1−β3ᵗ` 在数千步内都极小，除以它会把早期更新吹爆（论文图 27 的发散）。取而代之的是让缓冲慢慢充盈，同时由 **warmup 调度器**在前 `Warmup` 步内把它的影响力斜坡升起：α(t) 自 0 线性上升，β3(t) 按 *EMA 半衰期*线性插值（接近 1 时 β3 的固定增量远比接近 0.9 时重要）。
- `NewAdEMAMixDefault(lr, warmup)` 取 β = (0.9, 0.999, 0.9999)、α = 5、eps = 1e-8；论文把 warmup 设为总迭代数（`Warmup = 0` 关闭调度器）。
- 与本包的 Adam 一样，**不含解耦 weight decay**（论文的 λ / 官方的 `weight_decay`）：配方需要请自行施加。
- 一条固有不便，在长**零梯度暂停**（冻结的损失、数据空窗）之前务必知道：`g = 0` 期间，更新量按 `(β3/√β2)ᵗ` 演化——默认参数下约 1.0004ᵗ，即暂停期间*指数增长*直至触及 float32 极限。这是论文固有性质（不校正的慢速 EMA 除以衰减中的二阶矩），godoc 已注；避免长段零梯度，或接受暂停结束时的瞬态冲击。

### Schedule-Free AdamW：彻底不要学习率调度

Defazio 等，*The Road Less Scheduled*（[arXiv:2405.15682](https://arxiv.org/abs/2405.15682)，NeurIPS 2024 oral；MLCommons AlgoPerf 自调优赛道冠军）：以迭代平均取代 AdamW 的衰减调度——不需要终止时刻，不需要调衰减。每个参数跑三条序列；偏差校正跟随**当前官方实现**（schedulefree v1.3+，在根号内除以 `1−β2ᵗ`），而非论文初版变体：

```
y = (1−β1)·z + β1·x    在 y 处求梯度
z −= lr · g'           基础 AdamW 步，g' = g / (sqrt(v/(1−β2ᵗ)) + eps)（+ WeightDecay·y）
x = (1−c)·x + c·z      z 的多项式加权平均——可部署的权重
```

- **train/eval 契约——唯一必须做对的点。** 训练中每个参数的 `Data` 恒持 `y`（求梯度的点），绝不是可部署的 `x`。`Eval(params)` 原位转换到 `x`——评估、导出、为推理存档都在这里——`Train(params)` 在下一次 `Step` 之前转回。每个被 Step 过的参数自带**独立的模式位**，`Step` 在修改任何东西之前先检查全部模式位：对持有 `x` 的参数 Step 会按编号指名 panic——在 `x` 上训练是唯一会静默毁掉本方法的误用（官方实现也是在此抛错）。`Eval`/`Train` 幂等且只触碰已持有状态的参数；新建的优化器从 train 模式开始（构造时 `x = y = z`，官方强制的首个 `train()` 调用无可转换）。
- β1 = 0（Polyak-Ruppert 平均，`y = z`）被构造器**拒绝**——原位转换需要 `1/β1`。官方指引（godoc 已采纳）：LR 取调度版 AdamW 的 1×–10×（官方默认 0.0025），超长训练 β1 取 0.95–0.98。`WeightDecay` 为解耦衰减、在 `y` 处施加（默认 0），与上游一致。不要再叠加衰减调度——被取代的正是它。
- **检查点。** `SaveState` 持久化 `z`、`v`、计数器**以及每个参数的模式位**（kind 4，[persistence.md](persistence.md)）。train 模式存档用于逐位续训；**eval 模式**存档用于导出 `x` 或暂停：`LoadState` 之后所有被恢复的参数回到 eval 模式，`Step` 按编号指名 panic，`Train` 转换后续训——与同实例 Eval → Train → 续训路径逐位一致（`y→x→y` 往返本身有舍入，因此经 eval 检查点续训相对从未转换的轨迹差几个 ULP）。完整流程（含实测程序）见 [cookbook.md](cookbook.md) 食谱 14。

### 五者怎么选

朴素指引，不排座次：**SGD/Momentum** 用于理解循环（和玩具问题）；**Adam** 是默认的生产选择；**AdEMAMix** 适用于长程梯度记忆是任务要点的场景，同时接受 warmup 调度与零暂停告诫；**Schedule-Free** 适用于调不动或不想调衰减调度的场景，同时接受 train/eval 纪律与更大的 LR。本指南其余一切与优化器无关：四阶段循环、`ZeroGrad` 纪律、梯度裁剪、指针键状态、可热改的导出字段，以及经 `SaveState`/`LoadState` 的逐位续训。

## 为什么手写循环依然保留

本库的契约是一个小而可读、可审计的数值核心；一条更新规则就是五行 Go 代码，自己写出来，学习率调度、裁剪和正则化就都留在你的代码里——可见、可 diff，而不是藏在某个框架抽象背后。图引擎对叶节点如何更新不做任何假设。`optimizer` 包（见上）把这个循环恰恰原样打包为五条常用规则，是推荐的生产训练形态；手写版本依然是理解 `Step` 在做什么的基础——也是包未覆盖的一切更新规则（权重衰减变体、奇异调度）的去处。两种形态共享同一条纪律：裁剪仍由调用方负责（下一节）。

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

与 `optimizer.Step` 组合时裁剪方式相同，但要缩放梯度本身——`Step` 自己施加学习率，因此用 `float64` 累范数，超过 `maxNorm` 时把每个 `p.Grad.Data` 的全部元素乘以 `maxNorm/norm`，**在 `Step` 调用之前**完成：

```go
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
```

[persistence.md](persistence.md) 的「训练 → 保存 → 加载 → 续训」完整示例在一个完整程序里逐字演示了这种形态。

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
| 一步之后全部 `NaN` | `float32` 溢出扩散：一旦有一个元素是 `NaN`，非 MatMul 路径会把它带过整张图 | 裁剪；检查 `ts` 合理（以任务的采样间隔为锚——每步对应一个时间单位即 `ts = 1.0`；`ts ≥ 1e-3` 保持完整物理保真度——见 [ltc.md](ltc.md)） |
| 梯度恰好大了一倍 | 对同一张图调用了两次 `Backward`，或漏了 `ZeroGrad` | 每张新图一次 `Backward`；先对所有参数 `ZeroGrad` |
| 梯度有限但错误 | 参数 `Data` 在前向与反向之间被修改 | 更新严格放在 `Backward` 之后 |
| 缓慢蠕动、不收敛 | lr 过小，或损失的均值化掩盖了进展 | 用根目录 README 的快速上手程序做合理性检查 |
