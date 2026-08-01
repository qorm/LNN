> [English](../cfc.md) | 中文

# CfC 细胞：闭式连续时间

**摘要：** `nn/cfc.go` 的论文↔代码对照——闭式连续时间（Closed-form Continuous-time）细胞（Hasani 等，*Nature Machine Intelligence* 2022）：它沿用本库 LTC 所数值积分的同一条膜 ODE，但改用 Lemma 1 的闭式解（closed-form solution）在一步之内推进整个时间跨度，而不是跑 ODE 求解器循环——包括 exprel 稳定化、`ts` 契约、与 LTC 同构的参数化，以及相对官方 MLP 主干细胞的刻意取舍。

**读者对象：** 希望在代码中找到论文里每个符号的论文读者；在 `LTC` 与 `CfC` 之间做选择的工程师。

参考文献：R. Hasani, M. Lechner, A. Amini, D. Rus, R. Grosu,
[*Closed-form continuous-time neural networks*](https://doi.org/10.1038/s42256-022-00556-7)，
**Nature Machine Intelligence 4, 992–1003 (2022)**，
DOI [10.1038/s42256-022-00556-7](https://doi.org/10.1038/s42256-022-00556-7)，
arXiv [2106.13898](https://arxiv.org/abs/2106.13898)——注意发表期刊是 *Nature Machine Intelligence*，而非 *Nature Communications*。参考实现：[`mlech26l/ncps`](https://github.com/mlech26l/ncps)（`ncps/torch/cfc_cell.py`）与 [`raminmh/CfC`](https://github.com/raminmh/CfC)（`torch_cfc.py`）。

**命名警示：** 某些搜索引擎摘要把 CfC 这个缩写展开为 "liquid cubic activation"——它**不属于任何官方来源**。对两版论文（arXiv 预印本与 NMI 发表版）全文和两个官方仓库做六路 grep，"liquid cubic" 零命中——论文、代码、作者材料里都没有这个词。本实现不予采纳；别让摘要生成器把你带偏。

## ODE 正是 LTC 的那一条

CfC 保留 LTC 的膜 ODE——也就是 [ltc.md](ltc.md) 推导的那一条——写成能暴露闭式解的形式：

```
cm · dv/dt = −gleak·(v − vleak) + Σⱼ actⱼ·(erevⱼ − v)
           = −G·(v − A)

G = gleak + Σⱼ actⱼ                                   总电导
A = (gleak·vleak + Σⱼ actⱼ·erevⱼ) / G                 瞬时反转状态
actⱼ = softplus(wⱼ)·sigmoid(σⱼ·(v_pre − μⱼ))·maskⱼ    与 LTC 的激活逐字相同
```

`G` 与 `A` 聚合的恰恰是 LTC 的 `den`/`num` 所累加的电流（见 [ltc.md](ltc.md)），按单元逐元素成立。

## Lemma 1：闭式解

把激活在这一步内冻结——Lemma 1 近似：输入积分在当前输入处求值，`G` 与 `A` 因此分段常数——ODE 对 `v` 便是线性的，在时间跨度 `ts` 上有精确解（论文的 Theorem 1 / Lemma 1 / Eq. (8)）：

```
v_new = A + (v − A)·e^{−κ·ts},   κ = G/cm
      = v + (A − v)·F(B),        B = κ·ts,   F(B) = 1 − e^{−B}
```

`F ∈ [0, 1]`，因此 `v_new` 是旧状态 `v` 与瞬时反转状态 `A` 的凸组合：状态天然有界，完全不需要求解器展开（unfold）。代码采用第二种形式（`nn/cfc.go`，`Step`，225–268 行）：

| 量 | 代码 | 行号 |
|---|---|---|
| `G = gleak + Σ actⱼ` | `g := autograd.Add(gleak, autograd.Add(denS, denR))` | 252 |
| `A = (gleak·vleak + Σ actⱼ·erevⱼ) / (G + eps)` | `a := autograd.Div(…, autograd.Add(g, epsV))` | 254–257 |
| `B = κ·ts`，带溢出/符号钳制 | `b := c.decayRate(g, cm, epsV, ts)` | 259 |
| `F(B) = 1 − e^{−B}`，exprel 稳定化 | `f := c.decayFactor(b)` | 261 |
| `v_new = v + (A − v)·F` | `vNew := autograd.Add(h, autograd.Hadamard(autograd.Sub(a, h), f))` | 264 |

相对论文裸方程的一处诚实偏差：除数带保护，`κ = G/(cm + eps)`、`A = …/(G + eps)`，`eps = 1e-8`。在任何真实训练区制里，两个除数都被 softplus 正值性顶离零，保护在数值上不可见；它的存在只是为了让对抗性参数抽取不会制造 `Inf`/`NaN` 梯度。

## Algorithm 1：把 LTC 编译为闭式

论文的 Algorithm 1 把 LTC 逐突触编译成闭式更新，允许任意稀疏邻接。LNN 以 LTC 所用的同一套稀疏收缩（sparse contraction）与之同构：`drive()`（`nn/cfc.go:289-306`）为每个突触前神经元 `i` 构建一个 `[batch, units]` 激活块——以接线（wiring）掩码（mask）第 `i` 行门控的「列⊙行」外积——`contract`（`nn/cfc.go:322-338`）以 `+0` 播种的升序折叠（fold）收缩突触前轴：分母折叠各块，分子折叠按反转电位行加权的各块，二者都以与 units×units 单位阵的 MatMul 收尾。常量 `±1` 反转电位由共享 `erev`/`sErev` 存储的行视图常量承载（`erevRowViews`，与 `ltc.go` 共享），从不以叶节点入图。该收缩与 LTC 的 `contract` 逐行同构——同样四条约束、同一份逐位等价证明（见 [ltc.md](ltc.md) 稀疏收缩一节）——`±0` 角落也相同：全掩蔽突触后列落在 `+0` 而非旧 `Add` 链的 `−0`（下游不可观测，`(±0)² = +0`）；多源反向把零梯度归一为 `+0`；单源角落（`inDim = 1` 或 `units = 1`、零值梯度、`erev = −1`）可令零梯度携带 `−0`——值为 0 且不可观测，红队对 CfC 的全扫描未测出轨迹分歧。由于电位是行视图而非图叶，`erev`/`sErev` 完全不再入图——死梯度从结构上消失，而不仅仅为零。约定与二值接线掩码都沿用 LTC，因此 `NewCfC(inDim, units, wiring, rng)` 接受的 `Wiring` 拓扑与 `NewLTC` 完全相同（`nil` 即全连接）。

## 与 LTC 的关系：同一 ODE，两种积分器

`CfC` 与 `LTC` 离散化的是**同一条** ODE；差别在积分器：

| | `LTC` | `CfC` |
|---|---|---|
| 方案 | `unfolds` 个子步上的半隐式欧拉（semi-implicit Euler，见 [ltc.md](ltc.md)） | 解析积分器（analytical integrator）：一步走完 Lemma 1 闭式解 |
| 每个 RNN 步的图开销 | 随 `unfolds` 增长 | 与时间跨度无关的常量——没有子步循环 |
| 构造器 | `NewLTC(inDim, units, wiring, unfolds, rng)` | `NewCfC(inDim, units, wiring, rng)`——没有 `unfolds` |
| 其余一切 | 共享：13 个可训练张量、初始化区间、固定 ±1 且不在 `Parameters()` 中的反转电位、`ts` 契约、`Cell` 接口——`nn.Unroll` 驱动两者无需任何改动 |

由于参数抽取顺序一致，**相同 seed 给出逐位相同的初始化**（已验证：13 个参数外加 ±1 反转电位逐一相等）。

收敛性（红队独立 oracle 实测）：在固定时间跨度上，CfC 与 LTC 轨迹之差以**一阶** `p ≈ 1.0` 收敛——`ts` 减半，差距减半，符合同一 ODE 两种一阶离散化的预期。在*单步*上差值收缩得更快：`ts → 0` 时约 `O(ts²)`（`/tmp` 实测：log-log 斜率从 `ts = 0.4` 的 1.40 收紧到 `ts = 0.025` 的 1.95，与红队的 `p ≈ 1.89` 一致）。

**与官方实现的取舍。** 官方 CfC 细胞（`ncps`、`raminmh/CfC`）附带一个 MLP 主干变体：以 sigmoid 时间门后面的前馈 `ff1`/`ff2` 头作为闭式解的学习代理。本库的约定是突触 + 接线 + 固定 ±1 反转电位（见 [ltc.md](ltc.md)），因此这里的 CfC 实现的是论文的*方程级*闭式解——红队逐式审计的裁决是"比官方 pure 模式更贴方程"。如果你需要逐位复现上述仓库 MLP 主干的输出，这个细胞不是那个；如果你要的是以本库 LTC 参数化承载 ODE 的闭式解，那它就是。

## 与状态空间模型（SSM）的关系

CfC 更新步与选择性 SSM 更新步共享同一副骨架。取 Mamba 的 S6 的一个标量通道（Gu & Dao，[*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752)，arXiv:2312.00752）：递推式 `h_t = Ā·h_{t-1} + B̄·x_t`（该文 Eq. (2a)）经零阶保持离散化 `Ā = e^{Δa}`、`B̄ = (e^{Δa} − 1)·b/a`（该文 Eq. (4)，标量 `a`）后，在稳定区制 `a < 0` 下可改写为凸组合

```
h_t = e^{Δ(x)·a} · h_{t-1}  +  (1 − e^{Δ(x)·a}) · (−(b/a)·x_t)
```

——前一状态与输入稳态 `−(b/a)·x` 的凸组合；S6 又让 `Δ(x) = softplus(param + Linear(x))`（以及 `B`、`C`）成为输入的函数，即该文 3.2 节的选择机制（selection mechanism）。上文推导的 Lemma 1 更新步正是同一副骨架：

```
v_new = e^{−κ·ts} · v  +  (1 − e^{−κ·ts}) · A,     κ = G(x, v)/cm
```

对照本库的参数化，逐项对应：

| 选择性 SSM（Mamba，标量通道） | CfC（本库） |
|---|---|
| 衰减 `e^{Δ(x)·a} ∈ (0, 1)` | 衰减 `e^{−(G(x,v)/cm)·ts} ∈ (0, 1)` |
| 非负速率 `−Δ(x)·a`，依赖输入 | 非负速率 `G(x,v)/cm`——依赖输入**与状态**：突触门控 `actⱼ = softplus(wⱼ)·sigmoid(σⱼ·(v_pre − μⱼ))` 读取循环状态 |
| 目标 `−(b/a)·x_t`，即 `h′ = a·h + b·x` 的稳态 | 目标 `A = (gleak·vleak + Σⱼ actⱼ·erevⱼ)/G`，即膜 ODE 的稳态 |
| 步长折入学得/预测的 `Δ(x)` | 步长显式：每步由调用方给出 `ts`（可变、事件驱动） |
| 对角 `A`（每通道一个标量） | 逐单元对角动力学：单元之间经由速率与目标（`G` 与 `A` 中的突触求和）耦合，从不经由稠密转移矩阵 |
| 读出 `y_t = C·h_t` | 读出 `outW ⊙ v + outB`（逐单元仿射） |

也就是说，`G(x,v)/cm` 扮演的恰是 `Δ(x)·|a|` 的角色：两者都是非负、受输入调制的衰减速率，两个更新步都是在前一状态与当前输入的稳态之间插值。两处诚实的差异：CfC 的速率还依赖状态 `v` 本身——这使递推对 `v` 真正非线性，而 S6 的 `Δ`、`B`、`C` 仅是输入的函数；CfC 又把 `ts` 保持为显式量，因此同一个细胞无需重训即可处理非均匀采样或事件驱动的序列。

这一读法也是文献自身的走向，而非我们的私见。Liquid-S4（Hasani 等，[arXiv:2209.12951](https://arxiv.org/abs/2209.12951)，ICLR 2023）以线性 LTC 作为状态转移构建结构化 SSM——用作者的话说，一个"input-dependent state transition module"（依赖输入的状态转移模块）。LrcSSM（Farsang 等，[arXiv:2505.21717](https://arxiv.org/abs/2505.21717)，NeurIPS 2025）把 [arXiv:2403.08791](https://arxiv.org/abs/2403.08791) 的液阻-液容（LRC）动力学扩展成"a non-linear recurrent model that processes long sequences as fast as today's linear state-space layers"（一个以当今线性状态空间层的速度处理长序列的非线性循环模型），并明确把 Liquid-S4 与 Mamba 引为同属输入可变（input-varying）的系统。同一条研究谱系还延续到 Liquid AI 的 LFM2 端侧基础模型（[技术报告](https://arxiv.org/abs/2511.23404)，arXiv:2511.23404）——其混合主干把门控短卷积与分组查询注意力（grouped-query attention）块组合在一起。从这个角度看，本库的细胞就是可解释、变步长、稀疏接线的非线性 SSM 细胞——小到上表的每一项都能在代码里指认出来。

## exprel 稳定化

闭式连续时间著名的陷阱是 `B → 0` 处的裸商 `(1 − e^{−B})/B`：`1 − e^{−B}` 在有限精度下抵消为 0，再除以 `B` 得到垃圾（外加一条死梯度）。`decayFactor`（`nn/cfc.go:388-413`）以逐元素分支计算整个乘积 `F(B) = B·exprel(B)` 来绕开它：

| 分支 | 公式 | 理由 |
|---|---|---|
| `B < 1e-2` | 泰勒（Taylor）：`B − B²/2 + B³/6 − B⁴/24` | 舍去的余项 `≤ B⁵/120 < 8.3e-13`，远低于 `float32` 的 ε；`dF/dB = 1 − B + B²/2 − B³/6 → 1`，梯度在 `B → 0` 处存活 |
| `B ≥ 1e-2` | 原式 `1 − e^{−B}` | 此尺度下已无灾难性抵消；`B` 巨大时 `e^{−B}` 下溢为恰好的 0，`F` 饱和于 1（`v_new → A`） |

exprel 商里对 `B` 的除法在进入计算图之前就与外层的 `B` 因子**解析相消**——图中**根本没有除以 `B` 的节点**，自然无需防护。分支掩码是由 `B` 的数据构造的逐元素常量，梯度恰好流过激活的分支；两个分支在阈值处函数值与斜率都吻合到 `~1e-10`（红队以 8001 个点跨越 `1e-2` 扫描：跳变 `≤ 2.98e-8`；回归测试见 `TestCfCExprelBoundaryContinuity`）。

`B` 本身在上游 `decayRate`（`nn/cfc.go:351-364`）就受保护：时间缩放以 `float64` 计算、转换前钳制在 `1e30`，电导之比则套用 LTC 电容缩放所用的同一个光滑可微封顶（`cap(k) = k − softplus(k − hi)`），它同时保证 `B` 非负——负的衰减率会把 `e^{−B}` 变成爆炸。

## 时间跨度 `ts`

契约与 LTC 相同：`ts` 必须为正且有限；`NaN`、`±Inf`、零和负值都会 panic（`nn/cfc.go:228-230`）。极端处的行为：

| `ts` | 行为 |
|---|---|
| 正常区制（`ts ≳ 1e-3`） | 完整闭式保真度；封顶与裸代数逐位相同 |
| 微小（已测 `1e-40`） | `B ≈ 0`、`F ≈ B`，`v_new` 与 `v` 逐位相同——正确的 `dt → 0` 语义 |
| 巨大（已测 `1e300`） | 缩放钳制于 `1e30`，衰减因子（decay factor）`e^{−B} → 0`，状态弛豫到 `A`（稳态）；有限 |

## 参数表

与 LTC 同构（形状、初始化区间、softplus 约束全同——各区间的推导见 [ltc.md](ltc.md)）：

| 参数 | 形状 | 初始化 | 约束 | 角色 |
|---|---|---|---|---|
| `gleak` | `[units]` | U(0.001, 1) | softplus | 漏电导 |
| `vleak` | `[units]` | U(−0.2, 0.2) | 无约束 | 漏电反转电位 |
| `cm` | `[units]` | U(0.4, 0.6) | softplus | 膜电容（无 `unfolds/ts` 缩放——`ts` 经 `B = κ·ts` 进入） |
| `mu` | `[units, units]` | U(0.3, 0.8) | 无约束 | 循环突触激活中心 |
| `sigma` | `[units, units]` | U(3, 8) | 无约束 | 循环突触陡峭度 |
| `w` | `[units, units]` | U(0.001, 1) | softplus | 循环突触权重 |
| `sMu` | `[inDim, units]` | U(0.3, 0.8) | 无约束 | 感知突触中心 |
| `sSigma` | `[inDim, units]` | U(3, 8) | 无约束 | 感知突触陡峭度 |
| `sW` | `[inDim, units]` | U(0.001, 1) | softplus | 感知突触权重 |
| `inW`, `inB` | `[inDim]` | 1, 0 | 无约束 | 按特征的仿射输入映射 |
| `outW`, `outB` | `[units]` | 1, 0 | 无约束 | 按单元的状态→输出仿射映射 |
| `erev` | `[units, units]` | 随机 ±1 | **固定——不可训练** | 循环反转电位 |
| `sErev` | `[inDim, units]` | 随机 ±1 | **固定——不可训练** | 感知反转电位 |

`Parameters()` 返回与 LTC 相同的 13 个可训练张量（`nn/cfc.go:208-215`）；`erev`/`sErev` 因与 LTC 相同的结构性理由被排除——学习它们会翻转突触极性。

## 一个完整的训练循环

`CfC` 满足 `Cell` 接口，因此下面就是 `examples/ltc-sequence` 把细胞换掉、并去掉 ODE 子步循环的版本——训练使用 `optimizer` 包（见 [training.md](training.md)），梯度裁剪仍由调用方负责：

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

func main() {
	rng := rand.New(rand.NewSource(42))

	const (
		inDim   = 1
		units   = 8
		seqLen  = 12
		batch   = 16
		iters   = 250
		lr      = 0.05
		maxNorm = 1.0 // 全局梯度范数裁剪
		ts      = 1.0 // 每步的时间跨度
	)

	// 没有 unfolds 参数：闭式解一步推进整个时间跨度。
	cell := nn.NewCfC(inDim, units, nil, rng)
	readout := nn.NewLinear(units, 1, rng)
	params := nn.ParametersOf(cell, readout)
	opt := optimizer.NewSGD(lr)

	fmt.Printf("CfC accumulator task: inDim=%d units=%d seqLen=%d batch=%d, %d trainable tensors\n",
		inDim, units, seqLen, batch, len(params))

	var first, last float64
	for it := 0; it < iters; it++ {
		// 每轮迭代抽取全新的随机序列（在线 SGD）：
		// 有界累加器 s_t = clip(s_{t-1} + 0.25*u_t, -1, 1)。
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

		// 全局梯度范数裁剪（调用方负责，与 LTC 相同）：
		// 原地缩放梯度，然后让优化器走一步。
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

实际输出（Go 1.26，seed 42——确定性；每个 `loss` 在该轮更新*之前*测量）：

```
CfC accumulator task: inDim=1 units=8 seqLen=12 batch=16, 15 trainable tensors
iter   0  loss=0.620651
iter  50  loss=0.048169
iter 100  loss=0.041556
iter 150  loss=0.042028
iter 200  loss=0.021624
iter 249  loss=0.029091
first=0.620651 last=0.029091
```

损失从 `0.620651` 降到 `0.029091`（−95%），任务需要跨步记忆。

## 验证留痕

- **红队裁决：忠实且可信。** 对 NMI 发表版做方程级审计（Theorem 1 / Eq. (8) / Lemma 1 / Algorithm 1 逐一交叉核验），外加数值对抗 10/10 全过：8001 点跨阈值扫描（跳变 `≤ 2.98e-8`）、极端 `ts` 有限性（8/8）、掩码置零突触梯度恰为 0（9/9）、全参数 gradcheck 零失败。
- **库内回归测试**（`nn/cfc_test.go`）：`TestCfCGradcheckAllParameters`（全 13 参数有限差分检验，最大相对误差 `8.63e-3`——在 `float32` 中心差分噪声之内）、`TestCfCZeroMasksPureLeakClosedForm`（全零接线退化为纯泄漏闭式，`1e-4`）、`TestCfCDecayFactorExprelStability` 与 `TestCfCExprelBoundaryContinuity`（`1e-2` 分支边界）、`TestCfCStepTinyTsFixedPoint`（`ts = 1e-40` 时 `v` 逐位不动）、`TestCfCStepRejectsBadTs`（五类非法 `ts` panic）、`TestCfCDeterministicSameSeed`、`TestCfCParametersExcludeErev`。
- **`erev` 死梯度——已修复（阶段 8）：** `erev`/`sErev` 现已完全不再入图。±1 符号经由共享 `erev`/`sErev` 存储的行视图常量入图（阶段 8 最初以构造期指示矩阵（indicator matrix）承载；阶段 9 的稀疏收缩——见 [ltc.md](ltc.md) 稀疏收缩一节——以 `+0` 播种、末端单位阵 MatMul 的折叠（fold）取代了指示阵），因此 `erev`/`sErev` 字段是不带梯度的普通 `*tensor.Tensor`——死梯度从*结构上不可能*，而不仅仅为零；对字段类型做反射检查即可证实（`TestCfCReversalPotentialsCarryNoGradient`）。该手法与旧的 `Var` 叶驱动逐位等价（`TestCfCDriveBakeMatchesLegacyBitExact`；红队对多组 CfC 配置差分测试：前向与全 13 参数梯度逐位相同），且 `LoadCfC` 原位覆写 `erev`/`sErev` 存储——行视图与之共享存储，因此收缩**无需任何重建**即拾取流内极性（阶段 9 之前的设计在此处重新实体化稠密指示阵）——整谱符号翻转的 ±1 模式照样能加载，并且确实改变输出，这正是 `TestLoadCfCAcceptsFlippedReversalPattern` 所钉住的回归。
