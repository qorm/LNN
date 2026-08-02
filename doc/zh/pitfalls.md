> [English](../pitfalls.md) | 中文

# 陷阱、边界与残余风险

**摘要：** 本库*不*替你防护的一切，从红队审计提炼为用户须知：单线程契约、`float32` 溢出、重复 `Backward` 的精确语义、持久化层的不可信流契约，以及技术债路线图。

**读者对象：** 在 LNN 上交付任何东西之前，请读这一篇。

## 1. LNN 在设计上是单线程的

`Backward` 用朴素的 `+=`、无任何同步地修改叶变量的 `Grad` 缓冲区（`autograd/variable.go`）；张量直接暴露其 `Data`；`math/rand.Rand` 不是 goroutine 安全的。在共享参数的图上并发运行 `Backward` 是数据竞争（data race），会丢失或损坏梯度——已在 `go test -race` 下实证，这也是审计把并发从"bug"降级为"契约"的原因。

**为什么不直接加锁：** 本库用那种通用性换来了零同步开销、无隐藏耦合的内核；每个缓冲区都是你可以直接推理的朴素切片。

**受支持的并行模式：** 给每个 goroutine *它自己的*细胞、张量、变量和 RNG。绝不跨 goroutine 共享 `Variable`、`Tensor`、计算图——或 `rand.Rand`。以下范式已在 `-race` 下验证无竞争：

```go
var wg sync.WaitGroup
for g := 0; g < 4; g++ {
	wg.Add(1)
	go func(g int) {
		defer wg.Done()
		rng := rand.New(rand.NewSource(int64(g))) // 自己的 RNG
		cell := nn.NewLTC(1, 4, nil, 4, rng)      // 自己的参数
		x := autograd.Var(tensor.Uniform(rng, -1, 1, 2, 1))
		out, _ := cell.Step(x, nil, 0.1)          // 自己的图
		// ... 自己的 loss、自己的 Backward、自己的更新 ...
		_ = out
	}(g)
}
wg.Wait()
```

结果聚合（例如对 loss 求平均）必须由你自己的 channel/mutex 完成，作用于纯值，且在各 goroutine 完成反向传播之后。

## 2. float32 溢出没有全局防护

存储全是 `float32`；没有溢出检测，也没有 `float64` 模式。产生 `Inf`/`NaN` 的具体途径：

| 操作 | 溢出行为 |
|---|---|
| `Exp(x)` / `Pow(e, x)` | `x ≳ 88` 时得 `+Inf` |
| `Log(x)` | `x < 0` 得 `NaN`，`x = 0` 得 `-Inf` —— 定义域*不*检查 |
| `Div(a, b)` | `b = 0` 得 `±Inf`（普通 float32 除法语义） |
| 累加（`SumAll`、`MeanAll`） | 舍入漂移随长度增长；累加就是朴素的从左到右 `float32` 加法 |
| `Softplus(x)` | `x > 20` 时恰为 `x`（稳定分支），否则为 `log1p(exp)` |
| `SoftmaxRows`/`LogSoftmaxRows` | 内部已稳定化（减去最大值）；大 logit 安全 |

稳定化的内部实现只存在于上表所列之处（外加 `Sigmoid`、LTC 的 `ts` 缩放——见 [ltc.md](ltc.md)——以及 CfC 的 exprel 衰减因子（decay factor）与 `ts` 缩放，见 [cfc.md](cfc.md)）。其余都是你自己的问题：保持 logit 有界、钳制 `Log` 的输入、裁剪梯度（[training.md](training.md)）。一旦存在一个 `NaN`，它就会在本轮迭代的剩余时间里沿逐元素路径扩散。

**更糟的是，进入优化器状态的 `NaN` 是永久性的。** `Momentum` 与 `Adam` 的动量/二阶矩估计是滚动累加器：一旦有一个 `NaN` 梯度被折入其中，该参数的状态就永远是 `NaN`，之后每一步都把这份毒向前乘——红队的一次长跑（注入一个 `NaN`，60 轮迭代）全程保持全 `NaN`，始终没有恢复。关键在于：**后续的健康梯度无法把状态冲洗干净**，`ZeroGrad` 也无济于事：毒物存在于优化器的按参数状态缓冲里，而不在 `Grad` 里。被污染的优化器必须整个丢弃并重建（全新状态），或重置受影响的参数。这是"保持 logit 有界、裁剪梯度"（[training.md](training.md)）最有力的论据——一旦 `NaN` 进入有状态优化器，唯一的补救就是把那份状态推倒重来。

**一个需要知道的非对称 NaN 行为：** `MatMul` 在内层循环里跳过零乘数，而该跳过只检查**左**操作数（`tensor/ops.go:20`）。因此这个行为是有方向性的：`0 * NaN` 贡献 `0`（零乘数在乘积形成之前就被跳过），但 `NaN * 0` 仍得 `NaN`（`NaN` 左乘数不是零，于是乘积 `NaN * 0 = NaN` 被累加进去）：

```go
tensor.MatMul(tensor.FromData([]float32{0}, 1, 1),
	tensor.FromData([]float32{float32(math.NaN())}, 1, 1)).Data // [0]，而不是 [NaN]
tensor.MatMul(tensor.FromData([]float32{float32(math.NaN())}, 1, 1),
	tensor.FromData([]float32{0}, 1, 1)).Data // [NaN] —— 反过来就会毒化
```

不要指望 MatMul 能把稀疏位置上的 NaN 暴露出来，也不要以为这种跳零是对称的——它不是。

## 3. 对同一张图重复 Backward：严格线性

每轮迭代构建新图，只调用一次 `Backward`。如果对*同一张*图调用两次 `Backward`，叶梯度恰好再多一整趟反向传播——两次调用精确地得到两倍梯度，三次得到三倍（中间梯度在每次遍历后被清零，正是这一点让它保持线性而非超线性；修复前的 bug 会累积到 3 倍）：

```go
a := autograd.New([]float32{1, 2, 3}, 3)
y := autograd.SumAll(autograd.Hadamard(a, a)) // dy/da = 2a
y.Backward()
fmt.Println(a.Grad.Data) // [2 4 6]
y.Backward()             // 同一张图，第二次运行
fmt.Println(a.Grad.Data) // [4 8 12] —— 恰好翻倍
```

相关：在 `ZeroGrad` 之前，梯度会跨*不同的*图累积到共享的叶上——这正是训练循环依赖的特性。

## 4. 前向与 Backward 之间不要修改数据

少数反向步骤在反向时读取父节点的 `Data`（例如 `Log` 从父节点的*当前*值计算 `1/x`），而 `MatMul` 的反向使用保存的操作数张量。在前向与 `Backward` 之间修改叶节点会悄无声息地改变梯度：

```go
x := autograd.New([]float32{2}, 1)
l := autograd.SumAll(autograd.Log(x)) // d/dx = 1/x，在 Backward 时求值
x.Data.Data[0] = 8                    // Backward 之前的修改
l.Backward()
fmt.Println(x.Grad.Data) // [0.125] = 1/8，而不是 1/2
```

因此原地参数更新必须严格放在 `Backward` *之后*。

## 5. GatherRows 会拷贝索引（已修复的隐患）

`autograd.GatherRows(a, idx)` 在入口处拷贝 `idx`（`autograd/ops.go:891`），因此调用方可以在前向与反向之间随意复用或修改该切片——梯度由前向所用的索引算出：

```go
m := autograd.New([]float32{1, 2, 3, 4}, 2, 2)
idx := []int{0, 1}
out := autograd.GatherRows(m, idx)
idx[0], idx[1] = 1, 0 // 在 Backward 之前修改调用方的切片
autograd.SumAll(out).Backward()
fmt.Println(m.Grad.Data) // [1 0 0 1] —— 依然正确
```

这曾经是一个静默损坏 bug（闭包按引用捕获切片，红队实测到梯度被搅乱）；如今已有回归测试。

## 6. 微小 ts 只是有限性域

`LTC.Step` 接受任何正的有限 `ts`，但在 `ts ≈ 1e-38` 以下，`unfolds/ts` 缩放会溢出 `float32` 并被钳制：输出保持有限（已在 `1e-40` 和 `1e-300` 验证），但物理保真度已荡然无存。正常训练区制（`ts ≳ 1e-3`）与朴素 ODE 代数逐位相同。完整的表见 [ltc.md](ltc.md#时间跨度-ts)。

## 7. 随机初始化的细则

- **`Randn` 在 ≈ 7.43σ 处截尾。** Box-Muller 的 `u1` 均匀量被钳制在远离零的 `1e-12`（`tensor/random.go:35-36`），因此 `|sample| ≤ sqrt(−2·ln(1e-12)) ≈ 7.43` 恒成立——被省略的尾部质量约 1e-13。对权重初始化无关紧要；但**不要**把 `Randn` 当作通用正态采样器使用（例如依赖尾部事件的蒙特卡洛）。移除该钳制会改变固定 seed 的随机流，因此它被保留并文档化，而不是被修复。
- **`Uniform(lo, hi)` 在 `lo > hi` 时做镜像而不是 panic：** 值落在 `[hi, lo]`。为向后兼容而刻意保留的遗留行为；依赖区间的调用方应传 `lo ≤ hi`。
- 相同 seed ⇒ 逐位相同的随机流与模型（`TestLTCDeterministicSameSeed`）。

## 8. 形状约定不对称

`SumRows → [1, n]` vs `SumCols → [m]`，且 1D⊕1D 的结果被提升为 `[1, n]`。这些是 `nn` 和 `autograd` 内部所依赖的历史约定；经 v0.4.0 API 稳定窗口评估后**冻结**——统一它们只会让差分模糊测试（differential fuzzing）的证据基础全数失效，换取对称性而零性能收益。完整的决策留档、表和规避手段见 [shapes-and-broadcasting.md](shapes-and-broadcasting.md)。

## 9. 图就是内存模型

每个中间张量都保持存活，直到 `Backward` 完成。一次 LTC step 展开 `unfolds` 轮 ODE 迭代，自阶段 6 的突触向量化起每轮是 O(units) 个激活块（从 O(units²) 个逐突触节点降下来），自阶段 9 的稀疏收缩（sparse contraction）消灭稠密 `[units², units]` 指示矩阵（indicator matrix）起，突触前轴归约是 `+0` 播种、末端归一化 MatMul 收尾的折叠（fold）（见 [ltc.md](ltc.md)），阶段 7 的反向深改把逐节点分配数砍掉一半，阶段 8 的 Sigmoid–Hadamard 融合再进一步，阶段 10 的内嵌形状 backing（embedded backing，内联 `[4]int` 形状存储，每个张量少一次分配）再削一刀，阶段 16 的融合展开内核（每个 LTC step 一个 `FusedOp` 节点，对图路径逐位）把它带到 `UnrollBackward` **3,750 allocs/op**——较最初循环累计 −97%，`LTCStep` **77**（阶段 9 这一步曾使 allocs 上升约 30% 但墙钟下降约 13%，阶段 10 再降 allocs 约 23% 而墙钟持平，阶段 16 再砍墙钟约 2.1–2.3×，阶段 19 以感知入核与暂存复用再降 allocs 67%/44%——见 [architecture.md](architecture.md)）；`T` 步的序列会把这一切再乘以 `T`。内存随每次迭代的算子数增长，而不仅仅随参数增长——`units`、`unfolds` 和序列长度都要保持适度；或者改用 `CfC` 细胞（[cfc.md](cfc.md)），它的闭式步进没有 `unfolds` 因子；或者用 `nn.UnrollRemat` 把峰值图内存封顶在 O(chunk)（[architecture.md](architecture.md)；[cookbook.md](cookbook.md#13-长序列训练unrollremat-分块-bptt) 食谱 13）。

## 10. 持久化把模型文件当作不可信输入

`nn.SaveLTC`/`LoadLTC`、`SaveCfC`/`LoadCfC`、`SaveLinear`/`LoadLinear` 及其底层的 `serialize` 包，是本库"误用即 panic"契约的公开例外：检查点（checkpoint）恰是程序自己控制不了的输入，因此**加载路径上的一切失败都是 error，绝不 panic**，且恶意流的分配量只与它实际送达的字节数成正比。契约简述：

- **固定限额，先校验后分配：** 单个张量 ≤ `2^30` 个 float32（4 GiB 载荷），单条流 ≤ `2^20` 个张量，秩 ≤ `8` 轴，且 `LoadLTC` 在 blob 解析*之前*就拒绝 `unfolds > 1024`。元素计数用溢出安全乘法，因此声称维度宽达 `1<<62` 的流是一个 error，而不是一个 PB 级的 `make()`。
- **已知长度读端**（`bytes.Reader` 等）先拿每个载荷声明与剩余字节比对；**未知长度读端**（`io.Pipe`、`net.Conn`、`gzip.Reader`）采用渐进分配（progressive allocation）——一条声称 `2^30` 个元素却在 18 字节后停止的流，峰值约 33 KiB，以 `io.ErrUnexpectedEOF` 失败。
- **模型级校验：** kind 字节精确匹配（跨 kind 互载是指名道姓的错误）、掩码恰为 `{0, 1}`、反转电位恰为 `±1`（`NaN`/`±Inf`/`0`/小数一律拒绝）、张量数量精确，且一切形状在任何值被拷贝之前完成校验——失败的加载让目标分毫不动。
- **`serialize.LoadParameters` 保留陈旧 `Grad`：** 它原位覆写 `Data`（变量身份、从而图边得以存活），且刻意不动 `Grad`。在新图中复用加载后的变量之前先调用 `ZeroGrad`——与任何训练步之前完全一样。
- **`optimizer.SaveState`/`LoadState` 遵循同一纪律：** `"LNO1"` 状态流（state stream）先全验后应用（失败的加载让优化器逐位保持原样），只有 error 绝不 panic，恶意尺寸声明保持在实测的字节预算之内。保存状态后，续训（resume）与不间断训练逐位一致——见 [persistence.md](persistence.md) 优化器状态一节。

格式规格、API 指南与完整契约——包括版本规则（只读 version 1；未知版本报错而非误解析）——见 [persistence.md](persistence.md)。

## 残余留档（阶段 16 与 18）

阶段 16（融合 LTC 内核与 `UnrollRemat`）与阶段 18（融合 CfC 内核）工作的留档角落——选择披露而非修复；在文档所述用法下没有一项构成阻碍；体例照审计的留档表（问题/来源/严重度/处置）：

| # | 问题 | 来源 | 严重度 | 处置 |
|---|---|---|---|---|
| 1 | 融合 LTC 路径：**两个**操作数都是 NaN 的算子产出的 NaN，其载荷/符号位在融合路径与旧图路径之间不保真（NaN 位置集与一切有限值逐位一致）。融合 CfC 步（阶段 18）接受标准相同。溢出路径上**新产生**的 NaN（`Inf*0`）其符号位同样随架构与编译上下文而变——差分测试断言 NaN 位置集与有限值，从不断言 NaN 位模式（阶段 18 的 CI amd64 首跑实证了该角落） | 阶段 16 红队 | Low（接受） | 与单源符号位角落 F9-1 同一接受级——与训练无关、轨迹无可观测分歧；载荷取决于编译器的指令选择，不属于任何可复刻的舍入结构。契约见 `nn/ltc_fused.go` 与 `nn/cfc_fused.go` 头注的 NaN 语义节 |
| 2 | `UnrollRemat` 用于**逐步图结构依赖于张量值**的细胞时漂移约 1–2 ULP，单步结构探针不可探 | 阶段 16 红队 | Low（接受） | godoc 契约：`Step` 的图结构必须是 `(x, h)` 的纯函数；两种内置细胞均满足。值分支的自定义细胞在构造上即越出契约 |
| 3 | `UnrollRemat` 最坏情形：对抗性 loss 访问序（降序访问配小 chunkSize 为极端）迫使重算单元合并至 O(T)——峰值内存*和*算力均可超过一次全图反向 | 阶段 16 红队 | Info | 已写入 godoc 与食谱的 chunkSize 指引；把损失拼成升序访问各步输出（数据在前）即保 O(chunk) 快路径 |
| 4 | params 完备性审计的诚实性取决于细胞自报的 `Parameters()`：`Module` 若在自报列表中漏掉某个 `Step` 消费的叶，审计即失效，该叶会被静默多算（2–3 倍错梯度） | 阶段 16 红队 | Low（接受） | 两种内置细胞自报完备；对自定义细胞，`Parameters()` 的完备是细胞一侧的契约——审计覆盖细胞所披露的范围 |
| 5 | `nn` 覆盖率为 99.9% 而非 100%：唯一未覆盖语句是 `UnrollRemat` 单元扫描中一处构造性不可达的 nil-root 防火墙守卫 | 阶段 16 API 总扫 | Info | 作为防御未来单元切分改动的防火墙保留，不可达论证注释在 `nn/remat.go`；已在 README 状态表披露（照 optimizer 行 99.6% 的纪律） |
| 6 | 融合内核对**状态形状**严于图路径：可广播的错形 `h`（如 `[batch, 1]`、`[1, units]`）过去会被静默广播、产出形状错误的梯度；两个融合细胞现在都在入口处 panic | 阶段 18 红队 | Info（缺陷修复） | `Cell` 契约本就要求 `h` 为 `[batch, StateSize()]`——图路径的宽贷是缺陷，panic 才是契约。两内核同一裁决（LTC 在先，CfC 于阶段 18）；差分测试已钉住 |
| 7 | **反向 panic** 之后两路径行为不同：图路径会留下已投递的内部累加器残量（对受污染的图重试 `Backward` 会再次 panic），融合内核在任何投递之前 panic、重试保持干净 | 阶段 18 红队 | Info | 不复刻——干净的行为严格更优，且污染本就不可复现。指引与路径无关：`recover` 之后弃图重建 |

## 路线图与技术债

公开追踪；在文档所述范围（单线程、适度规模、合理的 `ts`/`lr`、调用方裁剪梯度）内，没有一项阻碍生产使用：

| 条目 | 状态 |
|---|---|
| `autograd.Div` 闭式化 | **已完成：** 单图节点 + 商法则反向（`da = g/b`、`db = −g·a/b²`，`autograd/ops.go:829-846`）。注意小除数固有的 `1/b²` 梯度放大依然存在——以 LTC 的 `eps = 1e-8` 下限计，最高约 `1e16` 倍——因此仍建议梯度裁剪 |
| LTC 突触向量化 | **已完成（阶段 6；收缩于阶段 9 重做）：** 掩码折叠出热路径；逐突触前神经元向量块（[ltc.md](ltc.md)）。阶段 6 以指示矩阵（indicator matrix）MatMul 收缩突触前轴（`LTCStep` 7,360 → 3,440 allocs/op、`UnrollBackward` 120,163 → 68,688）；阶段 9 以 `+0` 播种、末端单位阵归一 MatMul 收尾的稀疏折叠（fold）取代指示阵——与指示阵实现在正向和反向均逐位相同（对发布版 oracle 1,164 组差分 + 红队独立复测），而整 `Step` 相对*最初*的逐突触循环保持 ULP 级等价（前向 ≤ 1.79e-7、梯度 ≤ 1.19e-7）。阶段 9 使 allocs 上升约 43%/30%（折叠每级克隆），但 ns/op 下降约 21%/13%——墙钟净受益；阶段 10 的内嵌形状 backing（下两行）再降 allocs −18%/−23% 而墙钟持平。阶段 16 之前的数值：`LTCStep` 2,707 / `UnrollBackward` 31,994 allocs/op——已被下文阶段 16 融合行取代 |
| autograd 反向深改（闭包、融合、`addGrad` 克隆） | **已完成（阶段 7）：** 逐节点反向闭包改为 opKind 标签派发；`addGrad` 首次贡献所有权移交（所有权移交，ownership transfer；Clone 占比约 20% → 约 1%）；一元反向链融合（Sigmoid/Tanh 4→1 节点）、MatMul 转置缓冲消除、乘积-归约融合——全部以对重写前 oracle 的 52k 差分图逐位门禁，arm64 FMA 转换屏障保证融合循环保持两次舍入精确（[architecture.md](architecture.md)）。`UnrollBackward` 68,688 → 33,963 allocs/op（−50.55%）；另外四项基准 −23%…−58%，零回归。剩余的逐节点开销由下面的 `tensor.New` 一项追踪 |
| `tensor.New` 的逐节点固定开销（`Shape`/`Data` 双分配） | **已完成（v0.4.0，②内嵌 backing）：** 剖析显示剩余分配的 64.9% 是每个节点的前向输出及其 `Shape`/`Data` 双分配（实施前复测 60.4%）。v0.4.0 以**内嵌 `[4]int` 形状缓冲**（embedded backing，结构体内联 `shapeBuf`）消除 `Shape` 一半的堆分配：rank ≤ 4 零堆分配，超出则堆回退（保 `serialize` rank-8 流兼容）。五基准 allocs −18~−26%（每个 tensor 算子恰少一次 shape 分配），墙钟实测持平（±数% 噪声内）、字节 +3%——收益在分配次数与 GC 卫生，而非墙钟。选②内嵌 backing（约 10 个内部触点、零 API 破坏）而非①值类型形状字段（收益相同却破坏 233 处 `.Shape` 访问与 7 处直写）；新增导出的 `Tensor.Reshape` 取代 `Shape` 直写（负维度 panic）。余项 parents 定长槽化随后**亦完成（v0.4.3）：** 图节点 2 槽内联 + 变参溢出，五基准 allocs −10~−22%，对 15,360 图差分 fuzz 逐位等价——阶段 16 前最后一项性能留档就此关闭 |
| Sigmoid–Hadamard 融合反向（LTC 热路径模式） | **已完成（阶段 8）：** `autograd.SigmoidHadamard(z, w)` 把热路径的 `Hadamard(Sigmoid(z), w)` 融合成单节点（采纳于 `nn/ltc.go:423`）。前向逐位为构造性（逐字调用同一批 tensor 算子）；常规 2D 反向通过把 `g⊙w` 乘积在完全相同的位点舍入，达成与旧双节点链的逐位等价；异形或手设种子则逐字回退到旧组合（怪癖与 panic 契约一并保留）。`LTCStep` 2,442 → **2,306** allocs/op（−5.6%）、`UnrollBackward` 33,963 → **31,983**（−5.8%）——个位数收益，且为实测的结构上限：每个点位恰好省一个图节点加一个反向中间张量，剩余由 `tensor.New` 的逐节点固定开销主导（上一项）。见 [architecture.md](architecture.md) |
| CfC 的 `erev` 死梯度 | **已完成（阶段 8；指示阵于阶段 9 消灭）：** CfC 的反转电位不再入图——±1 符号由共享 `erev`/`sErev` 存储的行视图常量承载。字段是不带梯度的普通 `*tensor.Tensor`，因此死梯度从"为零"升级为**结构上不可能**；`drive()` 以与 LTC 相同的稀疏折叠（fold）收尾（`nn/cfc.go` 的 contract 与 LTC 逐行同构）。与旧的 `Var` 叶驱动逐位等价（红队差分测试），且 `LoadCfC` 原位覆写 `erev`/`sErev` 存储，行视图无需重建即拾取（[persistence.md](persistence.md)） |
| ~~指示矩阵 O(units³) 实体化~~ | **已完成（阶段 9，根因兑现）：** 稀疏收缩（sparse contraction，见 [ltc.md](ltc.md)）无论构造器还是加载路径都不再实体化 `[units², units]` 指示阵（indicator matrix）——`units = 1024` 全接线细胞的构造耗费约 32 MB（实测 36.4 MiB），而不再是旧的约 8 GiB 悬崖。`maxUnits`/`maxInDim` 加载上限按新的 O(units²) 内存模型重推：`256 → 2048`（上限处峰值 `92·U² B` ≈ 368 MiB——与旧制同一约 320 MiB 预算级，容量 8 倍），且最小攻击流峰值约为其送达字节的 1.5 倍，F1 契约（恶意流按送达字节比例分配）由根因兑现而非暂行封堵（[persistence.md](persistence.md)） |
| 优化器状态持久化 | **已完成（阶段 9）：** `optimizer.SaveState`/`LoadState`（`"LNO1"` 状态流）——续训与不间断训练逐位一致（50+50 vs 100 步，逐参数轨迹与 loss，三优化器全过）。不可信流纪律：先全验后应用、零副作用，只有 error 绝不 panic，恶意声明保持在实测字节预算内，Adam 更新计数受 `maxT = 2²⁴` 仅加载侧限额。格式规格与契约见 [persistence.md](persistence.md) |
| 统一归约形状约定（`SumRows`/`SumCols`、1D 提升） | **已冻结（v0.4.0）：** 评估实测了统一的真实代价——使 96k+52k+522 图差分模糊测试 oracle 全数失效、重做阶段 7 量级等价证明、重写 11 处 lift 点位 / 23 处守卫 / ≥17 个测试——换来的只有对称性，零性能收益；零用户窗口评估后择冻结。决策留档见 [shapes-and-broadcasting.md](shapes-and-broadcasting.md) |
| `tensor.Stack` | **已移除（v0.4.0）：** 库内调用 0、文档示例 0；产出没有其他算子消费的 3D 张量，为收窄公共面而删除（API 卫生） |
| CfC（Closed-form Continuous-time）细胞 | **已完成（阶段 6）：** `nn.CfC`（`nn/cfc.go`）——与 LTC 同一 ODE、同一套突触参数化，以 Lemma 1 闭式解驱动；论文↔代码对照与验证留痕见 [cfc.md](cfc.md)。新 API：仍可能演进 |
| 内置优化器 | **已完成（阶段 6）：** `optimizer` 包（SGD/Momentum/Adam，覆盖率 100%）；手写循环依然是理解引擎以及实现该包未覆盖规则的受支持范式——[training.md](training.md) 覆盖两种形态 |
| 序列化（Save/Load） | **已完成（阶段 7）：** `serialize` 包（带版本的 `"LNNS"` 张量流）外加 `nn` 的六个 Save/Load 函数（LTC/CfC/Linear）；对恶意流安全——只有 error 绝不 panic、固定限额先校验后分配、未知长度读端渐进分配（4 GiB 声明在 18 字节后停止仅峰值约 33 KiB）。红队变异模糊：7,500 个变异体 0 panic，资源耗尽加固后再测 1,200 个变异体，依然 0 panic。格式规格、API 指南与完整安全契约见 [persistence.md](persistence.md) |
| 基准/CI 工具 | **已完成：** `make bench`（18 项基准）+ GitHub Actions CI 双架构（ubuntu-latest + ubuntu-24.04-arm；gofmt 门禁、vet、build、`test -race`、example 冒烟、11 个原生 fuzz 目标冒烟） |
| LTC ODE 展开的图算子融合；重实体化 BPTT | **已完成（阶段 16）：** `autograd.FusedOp`（调用方自持反向闭包）+ LTC 展开融合为单图节点（`nn/ltc_fused.go`）——前向与梯度对融合前图路径逐位一致，公开 API 零变更。实测（`-benchtime=200x`）：`LTCStep` 约 203 µs / 2,122 allocs → **约 87 µs / 236**，`UnrollBackward` 约 2.8–3.3 ms / 28,273 → **约 1.33 ms / 6,662**（约 2.1–2.3×）。`nn.UnrollRemat` 提供分块 BPTT：梯度逐位一致，峰值图内存 O(chunk)（T = 512：保留约 0.65 MB 对比驻留约 8.3 MB）。留档角落见上文残余留档表；机制见 [architecture.md](architecture.md)，用法食谱见 [cookbook.md](cookbook.md#13-长序列训练unrollremat-分块-bptt) |
| CfC 闭式步融合；差分 fuzz 常驻化；CI 双架构 | **已完成（阶段 18）：** CfC 整个逐步子图融合为单个 `FusedOp` 节点（`nn/cfc_fused.go`）——`66 + 14·(inDim+units)` 个节点 → **任意维度 24 个**（`inDim=1, units=8` 时 190→24、`4, 16` 时 346→24），前向与梯度对图路径逐位一致（链式/堆叠含），公开 API 零变更、无新增导出符号；`h` 居父列表首位使 CfC 保持无脊柱类，`UnrollRemat` 维持 2F+1B 理想代价。实测（`-benchtime=200x`）：`CfCStep` 约 83.5 µs / 1,066 allocs → **约 34 µs / 52**（约 2.4×，allocs −95%），`UnrollRematCfC` 约 1.86 ms / 22,106 → **约 0.83 ms / 5,000**（约 2.2×，−77%）。三个逐位差分 fuzz 目标（`FuzzLTCFusedDifferential`、`FuzzCfCFusedDifferential`、`FuzzUnrollRematDifferential`；36 条种子语料）加入 fuzz 套件（8 → 11 目标），CI 全部六门禁在 amd64 与 arm64 双架构同跑。两处行为收紧（状态形状严格化、投递前 panic）见上文残余留档表 |
| 感知路径并入融合 LTC 内核；VJP/DFS/广播卫生批 | **已完成（阶段 19）：** 19a 内化感知驱动与 `numConst`/`denBase` 装配（每步约 `10·inDim+2` 个节点）——父列表 9 → 12，`hvN` 投递槽位重新推导，一个 Step 现在**任意维度 34 个节点**（15 算子 + 19 叶；此前 84、融合前 626；remat 折叠类不变，已钉住）。19b 让 VJP 的 18 块暂存面每次反向只分配一次（交付缓冲保持新分配——`addGrad` 所有权是逐位契约的一部分）；`Backward` 的 DFS 暂存池化（重入回退、panic 后 pristine、`TopoOrder` 结果恒新；`UnrollBackward` B/op −5.1%）；广播新鲜形状臂返回值类型 `[2]int`（3 → 2 allocs/op）。实测（`-benchtime=200x`）：`LTCStep` **约 78 µs / 77 allocs / 23 KB**（v0.5.2 约 87 µs / 236 / 47.7 KB），`UnrollBackward` **约 1.29 ms / 3,750 / 558 KB**（原约 1.33 ms / 6,662 / 927 KB），`UnrollRemat` **约 3.4 ms / 9,373**，`UnrollRematCfC` **约 0.84 ms / 4,829**；CfC 未触。零 API/行为变更（示例逐字一致、序列化字节一致） |
| ~~remat 折叠分类缓存~~ | **已否决并记档（阶段 19a C9）：** 不存在健全的缓存键——分类依赖于调用方每次重建的图*结构*，陈旧命中会静默跳过探针本应报告的多类 panic。与上文冻结约定同例否决；廉价的另一半已落地（两处 loss 图遍历共享一次 `TopoOrder`） |
