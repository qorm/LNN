> [English](../ltc.md) | 中文

# LTC 细胞：论文 ↔ 代码

**摘要：** 液态时间常数（Liquid Time-Constant）方程（Hasani 等，AAAI 2021）与 `nn/ltc.go` 的逐行对照，包括半隐式欧拉（semi-implicit Euler）代数推导、完整参数表、`ts` 契约和接线（wiring）语义。

**读者对象：** 需要信任（或修改）该细胞数值行为的工程师；希望在代码中找到论文里每个符号的论文读者。

参考文献：R. Hasani, M. Lechner, A. Amini, D. Rus, R. Grosu,
[*Liquid Time-constant Networks*](https://ojs.aaai.org/index.php/AAAI/article/view/17017)
（AAAI 2021），及参考实现 [`mlech26l/ncps`](https://github.com/mlech26l/ncps)。

## ODE

每个神经元都是一个具有输入依赖时间常数的漏电膜：

```
cm · dv/dt = −gleak · (v − vleak) + Σⱼ actⱼ · (erevⱼ − v)

    actⱼ = wⱼ · sigmoid( σⱼ · (v_pre,ⱼ − μⱼ) )
```

- `v` — 膜电位（隐藏状态），每个单元一个。
- `cm` — 膜电容；`gleak`/`vleak` — 漏电导及其反转电位（reversal potential）。
- 来自突触前源 `v_pre` 的突触（synapse）`j` 以一个*sigmoid 激活函数*激活，中心在 `μⱼ`、陡峭度 `σⱼ`，按权重 `wⱼ` 缩放；`(erevⱼ − v)` 是朝该突触反转电位的驱动力（`erev = +1` 为兴奋性，`−1` 为抑制性）。

`LTC` 类型的文档注释（`nn/ltc.go:15-29`）以论文的另一种形式 `dv/dt = −(1/tau + f(v, I))·v + f(v, I)·A` 陈述了同一个模型。

## 论文符号 → 代码行号对照（nn/ltc.go）

| 论文概念 | 代码 | 行号 |
|---|---|---|
| ODE 陈述 + 积分方案（文档） | `LTC` 类型注释 | 15–29 |
| 构造函数、校验 | `NewLTC` | 99–167 |
| 参数初始化区间 | `NewLTC` 字面量 | 132–160 |
| `eps`（分母保护）= `1e-8` | `ltcEps` 常量；`eps` 字段初始化 | 12–13；133 |
| 构造期图常量（折叠掩码、单位阵、+0 种子、反转电位行视图、突触项表） | struct 注释；`NewLTC` 字面量；`identityMat`；`erevRowViews`；`synapsePlan` | 50–86；152–159；184–190；199–206；208–264 |
| 可训练参数集（13 个张量） | `Parameters()` | 271–278 |
| 一次 RNN 步进 | `Step` | 284–333 |
| `ts` 契约检查（正且有限） | `Step` 防护 | 289–291 |
| 仿射输入映射 `inputs = x⊙inW + inB` | `Step` | 298 |
| 对 `cm, w, sW`（加 `gleak`）的 softplus 约束；掩码每 Step 一次折叠进权重 | `Step` | 303–306 |
| 感知突触电流（每步一次） | `synapses(inputs, sMu, sSigma, sWm, erevRowsS)` | 309 |
| Step 不变的循环参数行，切片一次 | `rows(c.mu/c.sigma/wM)`（助手 `rows` 在 369–375） | 313–315 |
| Step 常量膜项（`eps` 提升进 `denBase`） | `numConst`、`denBase` | 320–321 |
| ODE 展开（unroll）循环（`unfolds` 个子步） | `for t := 0; t < c.unfolds; t++` | 324–329 |
| 循环突触电流（每个子步重算） | `synapsesRows(v, muRs, sigRs, wmRs, erevRowsR)` | 325 |
| `num = cm_t⊙v + gleak⊙vleak + Σ actⱼ·erevⱼ` | `num := …` | 327 |
| `den = cm_t + gleak + Σ actⱼ (+ eps)` | `denBase` + `denR` | 321、328 |
| `v ← num / (den + eps)` | `v = autograd.Div(num, Add(denBase, denR))` | 328 |
| 仿射输出映射 `out = v⊙outW + outB` | `Step` | 331 |
| `cm_t = softplus(cm)·unfolds/ts`，带溢出安全缩放 | `scaledCapacitance` | 349–362 |
| 逐突触前神经元激活块 + 稀疏收缩 | `synapses` / `synapsesRows` / `contract` | 394–407 / 412–426 / 490–509 |

### 稀疏收缩

`synapses`/`synapsesRows`（`nn/ltc.go:394-426`）以每个突触前神经元一个向量块计算电流，不再是逐突触对的循环。每个参数矩阵的第 `i` 行仍然参数化*从*神经元 `i` *出发*的突触：

```go
// 每个突触前神经元 i 一个 [batch, units] 激活块
// （synapsesRows，nn/ltc.go:417-424）：
preCol := Col(pre, i)                              // [batch, 1]
z := Hadamard(sigRs[i], Sub(preCol, muRs[i]))      // σᵢⱼ·(v_pre,i − μᵢⱼ)
blocks[i] = SigmoidHadamard(z, wmRs[i])            // sigmoid(z) ⊙ wᵢⱼ·maskᵢⱼ，单个融合节点

// contract（nn/ltc.go:490-509）以 +0 种子折叠收缩突触前轴，
// 末端与 units×units 单位阵 ident 做 MatMul（正向为保值拷贝，
// 反向做归一化跳零）：
den = MatMul(fold(+0; b₀, b₁, …), ident)           // den[:,j] = Σᵢ blocksᵢ[:,j]
num = MatMul(fold(+0; b₀⊙erev₀, b₁⊙erev₁, …), ident)
                                                   // num[:,j] = Σᵢ blocksᵢ[:,j]·erev[i,j]
```

相对最初的逐突触循环，有两处结构性变化源自阶段 6 并保留至今：**掩码移出热路径**——接线掩码以每 Step 一次矩阵 Hadamard 折叠进正值约束后的权重，`wm = softplus(w)⊙mask`（`nn/ltc.go:305-306`），而不是每个子步每个突触一次带掩码的乘法；**循环参数行每 Step 只切一次**——`mu`、`sigma` 与掩码后的权重矩阵都是 Step 不变量，`rows`（`nn/ltc.go:369-375`）切片一次供展开循环复用，使每个矩阵 `units·(unfolds−1)` 个 `SliceRow` 节点留在图外。

第三处变化——本节因而得名——来自阶段 9。突触前轴的归约过去是对稠密 `[pre·units, units]` 指示矩阵（indicator matrix）做 MatMul（构造期构建）：那会把 O(units³) 个 float32 实体化——`NewLTC(4, 1024, …)` 要分配约 8 GiB 指示阵——在构造器与加载路径上都构成内存悬崖（红队总扫的 F1；加载侧在阶段 8 由 `maxUnits` 封顶暂行封堵，见 [persistence.md](persistence.md)）。**稀疏收缩**（sparse contraction）取代了指示阵：归约现在是一个从标量 `+0` 播种的**折叠**（fold）——按升序从左到右累加激活块，分子各项乘以反转电位行视图——末端与单位阵做一次 MatMul。持久收缩状态为 `ident`（一个 units×units 矩阵）、`zeroV`（一个标量）、`erevRows*` 行视图（与 `erev`/`sErev` 共享存储——不占额外浮点）、以及 `planR`/`planS` **项表**（term table）：合计 O(接线非零数) 个 `int32` 元数据、总量 O(units²)——且**任何 Step 都不实体化 `[units², units]` 张量**。本机实测：`NewLTC(4, 1024, nil, 6, …)` 总分配 **36.4 MiB**、`NewCfC(4, 1024, nil, …)` **32.4 MiB**，对照旧制 8 GiB 悬崖（红队复测一致：36.4/32.4 MiB；门禁由 `TestLTCSparseContractionLargeCellMemoryGate` 钉住）。

**项表语义。** `synapsePlan`（`nn/ltc.go:208-264`）按突触后神经元 `j` 记录使 `mask[i, j] == 1` 的突触前下标 `i` 的升序列表——恰是收缩所求和的 (i, 系数) 项表（项 (i, j) 的分子系数为 `erev[i, j]`，分母系数为 1）。存储为 O(nnz) 个 `int32` 加每个突触后神经元一个切片头：units=1024 全连接时为 4 MiB 项表，对照 8 GiB 指示阵。项表*记录并计数*收缩的各项（`terms()` 即掩码非零数）；折叠本身按整个突触前行遍历而非逐项表行走，因为未接线突触在算术上是中性的：被掩码置零的块表项恰为 `+0`（sigmoid ≥ 0 乘以置零掩码后的权重），而加 `+0` 是恒等操作，因此全行折叠精确求和已接线的各项，逐位不差。

**逐位等价：四约束。** 该归约与其所替换的指示阵 MatMul 逐位相同——正向**与**反向皆然——因为它精确复刻了那个 MatMul 的四条定义性行为（完整证明在 `contract` 的文档注释，`nn/ltc.go:428-509`；红队已验证四约束在代码中为真，而非仅在注释里）：

1. **升序突触前次序。** MatMul 第 `j` 列按升序 `i` 累加其非零元；折叠按下标序遍历各块，因此每一列 `j` 以完全相同的次序累加完全相同的 (i, 系数) 对（未接线突触不在项表却在折叠里，按上面的 `+0` 中性论证不改变求和）。
2. **操作数次序。** 分子的逐项乘积是 block·erev——激活在左、系数在右——与 MatMul 求值 `av·brow[j]` 完全一致。
3. **跳零。** MatMul 跳过为零的左操作数，含 −0（`tensor.MatMul` 的 `av==0` 分支——F-RT1 `+0` 行为的来源，刻意保留）。正向上，折叠以算术复刻该跳过：零块使该项为 ±0，而以 +0 播种、舍入到最近偶数的求和永远产生不出 −0，故被跳过与被加上的零落在相同位模式。反向上，末端与 `ident` 的 MatMul 运行真正的 `tensor.MatMulTransB`，其 `av==0` 分支把零入梯度归一为 `+0` 贡献——该宣称的确切覆盖面见下面的诚实角落。
4. **+0 累加器。** 折叠以 `zeroV` 标量 `+0` 播种，复刻 MatMul 全新填零的输出缓冲——这使全掩蔽的突触后列保持在 `+0`（`0x00000000`）而非 `−0`（由 `TestLTCSparseContractionFullyMaskedColumnZeroSign` 钉住）。

实施方差分门禁（1,164 组 × `Float32bits`；oracle = 从 git 提取的指示阵发布版实现：LTC/CfC × 全连与稀疏接线 × 多种规模/时间步/状态，含 12 组反向跳零对抗角）**零差异**；红队独立复测（522 组刻意异源分布：前向 + 全 13 参数 + 叶梯度逐位相同）相互印证。CfC 的 contract 与 LTC 逐行同构（`nn/cfc.go:307-323`）。

**等价性在 Step 层面是 ULP 级的——不是逐位。** 孤立的稀疏驱动与最初的逐突触循环逐位相同（`nn/ltc_test.go` 的 `TestLTCSynapsesVectorizedEquivalence` 以严格 `==` 回归测试：mask ∈ {0,1} 加上升序折叠次序精确复现旧 `Add` 链，舍入次序无任何变化）——但有一个诚实的 `±0` 符号位角落，它有**三层**：

- **正向，全掩蔽的突触后列。** 那里每个 `actⱼ` 都是 `+0`；最初的 `Add` 链累加 `+0 · erevⱼ`，在 `erevⱼ = −1` 处得 `−0`（`0x80000000`）；而收缩——旧指示阵 MatMul 与新的 +0 播种折叠皆然——让累加器停在 `+0`。两者在 `==` 下相等，且差异在下游不可观测：平方误差损失里 `(±0)² = +0`。
- **反向，多源路径。** 突触前行多于一个时，den 与 num 都以归一化 MatMul 收尾：其 `av==0` 分支把零入梯度归一为 `+0` 贡献——与旧指示阵 MatMul 反向所算逐位相同（num 的 Hadamard 反向若贡献 −0，也会与 den 的 +0 相加得 `(+0)+(−0) = +0`）。
- **反向，单源角落。** 突触前行恰有一个（感知路径 `inDim = 1`，或循环路径 `units = 1`）时，den 走历史捷径——裸块，无折叠、无归一化 MatMul。此时若入梯度值恰为零且 `erev = −1`，num 折叠的 Hadamard 反向算出 `(+0)·(−1) = −0` 而无归一化 MatMul 洗去符号，于是某些零梯度携带 `−0` 符号位（红队 F9-1 实测：522 组差分中 4 组命中此角，值全为 0）。这同样下游不可观测：`p − LR·(±0)` 对任何本身非 `−0` 的 `p` 逐位相同，Momentum/Adam 累加器归一 `±0`，红队实测训练轨迹零分歧——一个**刻意接受的符号位角落**（修补它要为单源路径付出每步一次 MatMul 的永久代价，换取任何计算都不可观测的符号位；CfC 的同构 contract 存在同一结构性角落，其全扫描未测出分歧）。

但整个 `Step` 把 `eps` 和 Step 常量项（`gleak⊙vleak`、感知电流）提升到展开循环之外，相对*最初*的逐突触循环改变了 `float32` 的结合次序。红队的独立 oracle——从 git 历史提取重写前的 `ltc.go`，在 13 组随机化配置上差分测试——实测最大差：前向 **1.79e-7**、全参数 BPTT 梯度 **1.19e-7**：ULP 级、良性，但不是"逐位"。

**实测开销。** 当前值（同机，`-benchtime=200x`，复验过）：`LTCStep` **236 allocs/op**、`UnrollBackward` **6,662 allocs/op**——较最初逐突触循环（7,360 / 120,163）累计 **−97%/−94%**（阶段 6 向量化降至 3,440 / 68,688，阶段 7 反向深改把反向降至 33,963，阶段 8 Sigmoid–Hadamard 融合再降至 2,306 / 31,983，阶段 10 内嵌形状 backing（embedded backing）降至 2,707 / 31,994——allocs 再降 −18%/−23% 而墙钟持平，阶段 16 融合展开内核降至当前值——见 [architecture.md](architecture.md)）。阶段 9 这一步的诚实披露：稀疏收缩使 allocs 相对阶段 8 **上升**约 43%/30%（折叠每级的 opAdd b 支克隆）——工单预期"中性或略降"错了——但 **ns/op 下降约 21%/13%**（红队独立复测），因消灭了稠密指示阵 MatMul 对零行的 O(units³) 空转内层循环与反向大分配。**分配次数换走了无用算力，墙钟净受益**。example 的首轮 loss 仍是 `0.690761`，与重写前逐位一致。

## 半隐式欧拉的推导

参考实现以半隐式（后向）欧拉步长积分该 ODE：漏电与突触驱动力在*新*电压 `v_{k+1}` 处求值，这使更新成为一个精确的除法而非指数运算。设子步长 `dt = ts/unfolds`：

```
cm · (v_{k+1} − v_k)/dt = −gleak·(v_{k+1} − vleak) + Σⱼ actⱼ·(erevⱼ − v_{k+1})
```

把 `v_{k+1}` 收集到左边：

```
(cm/dt)·v_{k+1} + gleak·v_{k+1} + (Σⱼ actⱼ)·v_{k+1}
        = (cm/dt)·v_k + gleak·vleak + Σⱼ actⱼ·erevⱼ
```

```
            (cm/dt)·v_k + gleak·vleak + Σⱼ actⱼ·erevⱼ      num
v_{k+1}  =  ───────────────────────────────────────────  = ───────
             cm/dt + gleak + Σⱼ actⱼ                        den
```

代码逐元素计算 `v ← num / (den + eps)`（`nn/ltc.go:327-328`），其中 `cm/dt = softplus(cm)·unfolds/ts` 由 `scaledCapacitance` 构建。所有量都是按单元的向量，因此每个运算都是逐元素或广播；退化情形（所有接线掩码为零）精确约化为一个漏电积分器 `v ← (a·v + b·vleak)/(a + b + eps)`，其中 `a = softplus(cm)·unfolds/ts`、`b = softplus(gleak)`——由 `nn/ltc_test.go` 中的 `TestLTCZeroMasksLeakyIntegrator` 以闭式做了回归测试。

因为求解对 `v` 是隐式的，该更新在大 `ts` 下是稳定的（状态朝稳态弛豫而不是爆炸），这正是该细胞的可变步长区制可用的原因。

## 参数表

所有区间遵循 ncps 参考实现。"Softplus"表示原始参数不受约束，以 `softplus(raw)` 进入 ODE——即参考实现的 `implicit_param_constraints` 模式——无需优化器侧裁剪即保证正值性。

| 参数 | 形状 | 初始化 | 约束 | 角色 |
|---|---|---|---|---|
| `gleak` | `[units]` | U(0.001, 1) | softplus | 漏电导 |
| `vleak` | `[units]` | U(−0.2, 0.2) | 无约束 | 漏电反转电位 |
| `cm` | `[units]` | U(0.4, 0.6) | softplus，按 `unfolds/ts` 缩放 | 膜电容 |
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

`Parameters()` 返回这 13 个可训练张量；`erev`/`sErev` 被刻意排除在外（`nn/ltc.go:271-278`）。学习反转电位会让突触在兴奋性与抑制性极性之间翻转，把 LTC 退化为普通可塑网络——±1 的符号模式是结构性的，在构造时抽取一次。

与参考实现的已知偏差：`inW`/`outW` 恰好初始化为 1、`inB`/`outB` 为 0，而 ncps 使用 U(0.9, 1.1)/U(−0.1, 0.1)。两个映射都可训练，差异在训练中会被冲掉；只影响第 0 步。

## 时间跨度 `ts`

`Step(x, h, ts)` 在 `ts` 个单位的时间跨度上积分 ODE，分 `unfolds` 个子步（`dt = ts/unfolds`）。这是网络"液态"的部分：由调用方驱动时间，事件驱动、逐步可变——小 `ts` 几乎不推进膜，大 `ts` 让它们弛豫向稳态。它对应 ncps 的 `elapsed_time`。

**契约：`ts` 必须为正且有限。** `NaN`、`+Inf`、`-Inf`、零和负值都会 panic（`nn/ltc.go:289-291`）：

```go
_, _ = cell.Step(x, nil, 0.01)  // 没问题：快速动态
_, _ = cell.Step(x, nil, 10.0)  // 没问题：接近稳态
cell.Step(x, nil, math.Inf(1))  // panic：无限时间跨度被拒绝
cell.Step(x, nil, math.NaN())   // panic
```

实现的有限性域（`scaledCapacitance`，`nn/ltc.go:349-362`）：

| `ts` 区间 | 行为 |
|---|---|
| `ts ≳ 1e-3`（任何真实训练区制） | 溢出防护与朴素 `softplus(cm)·unfolds/ts` 逐位相同——对 `ts ∈ [1e-3, 100]` 的扫描实测最大差异为 `0`；完整物理保真度 |
| 低至 ≈ `1e-38` | 仍是真实的 ODE；与朴素公式的首次位级偏差出现在 `ts ≈ 1e-37`–`1e-38` 附近（unfolds=6 实测：`1e-36` 无偏差，`1e-37` 出现偏差；硬性溢出钳制始于 `ts < unfolds/MaxFloat32`，unfolds=6 时约 `1.76e-38`），远低于任何调度 |
| `0 < ts ≲ 1e-38` | `unfolds/ts` 超过 `MaxFloat32`；缩放被钳制，电容乘积由一个光滑可微的 min 封顶。输出保持**有限**（已在 `ts = 1e-40` 和 `1e-300` 验证），但这只是有限性域，而非物理域——没有合理的调度会到这里 |
| 巨大 `ts`（已测 `1e300`） | 缩放 → 0，状态弛豫到稳态；有限 |

## 接线

`Wiring` 用二值掩码（mask）对单个突触做门控（`nn/wiring.go`）：

- **感知掩码** `[inDim, units]` — 从输入到神经元的突触；
- **循环掩码** `[units, units]` — 神经元之间的突触；
- 表项 `[i, j]` 是**从**输入/神经元 `i` **到**神经元 `j` 的突触（参数矩阵的第 `i` 行参数化突触前神经元 `i`）。

构造器：

| 构造器 | 语义 |
|---|---|
| `FullyConnected(inDim, units)` | 每个突触都存在（全 1）；`NewLTC` 收到 `nil` 时的默认值也是如此 |
| `RandomSparse(inDim, units, sensoryP, recurrentP, rng)` | 每个突触以概率 `p` 独立存在（按表项的伯努利分布）。不保证连通性：`p` 较低时神经元可能孤立。概率必须落在 `[0, 1]` 内——`NaN`、负值和 `>1` 的值会 panic，维度 `< 1` 同样 panic。 |

掩码在构造后不可变：字段不导出，全掩码访问器 `Sensory()`/`Recurrent()` 返回深拷贝，而 `Step` 内部热路径上的行访问器直接读取原始掩码。红队已验证：篡改任何返回的张量，细胞输出依然逐位相同。

一个稀疏细胞，在多个时间尺度上驱动（seed 42，确定性）：

```go
rng := rand.New(rand.NewSource(42))
wiring := nn.RandomSparse(2, 6, 0.8, 0.5, rng) // 感知 11/12、循环 16/36 个突触
cell := nn.NewLTC(2, 6, wiring, 6, rng)        // 13 个可训练参数张量

x := autograd.Var(tensor.Uniform(rng, -1, 1, 3, 2))
out, h := cell.Step(x, nil, 0.1) // batch 3，零初始状态
_, h = cell.Step(x, h, 1.0)      // 传递状态；更大的 ts -> 弛豫得更远
ys, hN := nn.Unroll(cell, []*autograd.Variable{x, x, x}, nil, 0.1)
```

## 与 ncps 参考实现的关系

| ncps 概念 | LNN 对应 |
|---|---|
| 带 `units`、`unfolds`（默认 6）的 LTC 层 | `NewLTC(inDim, units, wiring, unfolds, rng)` |
| ODE 求解器：`unfolds` 个子步上的半隐式欧拉 | 相同方案（`Step` 循环，`nn/ltc.go:324-329`） |
| `implicit_param_constraints`（softplus 正值性） | 对 `cm`、`gleak`、`w`、`sW` 的 softplus |
| 参数初始化区间 | 原样采用（上表） |
| 来自接线的固定 ±1 反转电位 | `erev`/`sErev`，不在 `Parameters()` 中 |
| 每步 `elapsed_time`（可变步长/事件驱动训练） | `Step`/`Unroll` 的 `ts float64` 参数（选 float64 是为了与 ncps 一致，并让 `unfolds/ts` 对微小 `ts` 保持安全） |
| 感知 vs 循环突触划分 | `sMu/sSigma/sW/sErev` vs `mu/sigma/w/erev` + 接线掩码 |
| CfC（闭式连续时间）细胞 | **已实现**为 `nn.CfC`（`nn/cfc.go`）：同一条 ODE、同一套 13 参数突触参数化，改用 Lemma 1 闭式解推进而非本欧拉循环——见 [cfc.md](cfc.md) |
