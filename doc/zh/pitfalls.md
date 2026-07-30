> [English](../pitfalls.md) | 中文

# 陷阱、边界与残余风险

**摘要：** 本库*不*替你防护的一切，从红队审计提炼为用户须知：单线程契约、`float32` 溢出、重复 `Backward` 的精确语义、持久化层的不可信流契约，以及技术债路线图。

**读者对象：** 在 lnn 上交付任何东西之前，请读这一篇。

## 1. lnn 在设计上是单线程的

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

`autograd.GatherRows(a, idx)` 在入口处拷贝 `idx`（`autograd/ops.go:855`），因此调用方可以在前向与反向之间随意复用或修改该切片——梯度由前向所用的索引算出：

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

`SumRows → [1, n]` vs `SumCols → [m]`，且 1D⊕1D 的结果被提升为 `[1, n]`。这些是 `nn` 和 `autograd` 内部所依赖的历史约定；改动它们是 API 破坏性变更，被单独追踪（路线图）。完整的表和规避手段见 [shapes-and-broadcasting.md](shapes-and-broadcasting.md)。

## 9. 图就是内存模型

每个中间张量都保持存活，直到 `Backward` 完成。一次 LTC step 展开 `unfolds` 轮 ODE 迭代，自阶段 6 的突触向量化起每轮是 O(units) 个向量块加两次 MatMul 收缩（从 O(units²) 个逐突触节点降下来——见 [ltc.md](ltc.md)），阶段 7 的反向深改又把逐节点分配数砍掉一半，阶段 8 的 Sigmoid–Hadamard 融合再进一步（`UnrollBackward` 31,983 allocs/op，较最初循环累计 −73%——见 [architecture.md](architecture.md)）；`T` 步的序列会把这一切再乘以 `T`。内存随每次迭代的算子数增长，而不仅仅随参数增长——`units`、`unfolds` 和序列长度都要保持适度；或者改用 `CfC` 细胞（[cfc.md](cfc.md)），它的闭式步进没有 `unfolds` 因子。

## 10. 持久化把模型文件当作不可信输入

`nn.SaveLTC`/`LoadLTC`、`SaveCfC`/`LoadCfC`、`SaveLinear`/`LoadLinear` 及其底层的 `serialize` 包，是本库"误用即 panic"契约的公开例外：检查点（checkpoint）恰是程序自己控制不了的输入，因此**加载路径上的一切失败都是 error，绝不 panic**，且恶意流的分配量只与它实际送达的字节数成正比。契约简述：

- **固定限额，先校验后分配：** 单个张量 ≤ `2^30` 个 float32（4 GiB 载荷），单条流 ≤ `2^20` 个张量，秩 ≤ `8` 轴，且 `LoadLTC` 在 blob 解析*之前*就拒绝 `unfolds > 1024`。元素计数用溢出安全乘法，因此声称维度宽达 `1<<62` 的流是一个 error，而不是一个 PB 级的 `make()`。
- **已知长度读端**（`bytes.Reader` 等）先拿每个载荷声明与剩余字节比对；**未知长度读端**（`io.Pipe`、`net.Conn`、`gzip.Reader`）采用渐进分配（progressive allocation）——一条声称 `2^30` 个元素却在 18 字节后停止的流，峰值约 33 KiB，以 `io.ErrUnexpectedEOF` 失败。
- **模型级校验：** kind 字节精确匹配（跨 kind 互载是指名道姓的错误）、掩码恰为 `{0, 1}`、反转电位恰为 `±1`（`NaN`/`±Inf`/`0`/小数一律拒绝）、张量数量精确，且一切形状在任何值被拷贝之前完成校验——失败的加载让目标分毫不动。
- **`serialize.LoadParameters` 保留陈旧 `Grad`：** 它原位覆写 `Data`（变量身份、从而图边得以存活），且刻意不动 `Grad`。在新图中复用加载后的变量之前先调用 `ZeroGrad`——与任何训练步之前完全一样。

格式规格、API 指南与完整契约——包括版本规则（只读 version 1；未知版本报错而非误解析）——见 [persistence.md](persistence.md)。

## 路线图与技术债

公开追踪；在文档所述范围（单线程、适度规模、合理的 `ts`/`lr`、调用方裁剪梯度）内，没有一项阻碍生产使用：

| 条目 | 状态 |
|---|---|
| `autograd.Div` 闭式化 | **已完成：** 单图节点 + 商法则反向（`da = g/b`、`db = −g·a/b²`，`autograd/ops.go:793-810`）。注意小除数固有的 `1/b²` 梯度放大依然存在——以 LTC 的 `eps = 1e-8` 下限计，最高约 `1e16` 倍——因此仍建议梯度裁剪 |
| LTC 突触向量化 | **已完成（阶段 6）：** 掩码折叠出热路径；逐突触前神经元向量块 + 两次构造期指示矩阵（indicator matrix）MatMul 收缩（[ltc.md](ltc.md)）。`LTCStep` 7,360 → 3,440 allocs/op（−53.3%）、`UnrollBackward` 120,163 → 68,688（−42.8%）。整 `Step` 与重写前循环为 ULP 级等价（前向 ≤ 1.79e-7、梯度 ≤ 1.19e-7，红队独立 oracle）；逐位一致仅对孤立的 `synapses()` 驱动成立 |
| autograd 反向深改（闭包、融合、`addGrad` 克隆） | **已完成（阶段 7）：** 逐节点反向闭包改为 opKind 标签派发；`addGrad` 首次贡献所有权移交（所有权移交，ownership transfer；Clone 占比约 20% → 约 1%）；一元反向链融合（Sigmoid/Tanh 4→1 节点）、MatMul 转置缓冲消除、乘积-归约融合——全部以对重写前 oracle 的 52k 差分图逐位门禁，arm64 FMA 转换屏障保证融合循环保持两次舍入精确（[architecture.md](architecture.md)）。`UnrollBackward` 68,688 → 33,963 allocs/op（−50.55%）；另外四项基准 −23%…−58%，零回归。剩余的逐节点开销由下面的 `tensor.New` 一项追踪 |
| `tensor.New` 的逐节点固定开销 | 后续方向：剖析显示剩余分配的 64.9% 是每个节点的前向输出及其 `Shape`/`Data` 双分配；进一步压缩需要 parents 定长槽化（受阻于既有结构断言测试）与 Tensor 定秩 Shape（公共 API 破坏） |
| Sigmoid–Hadamard 融合反向（LTC 热路径模式） | **已完成（阶段 8）：** `autograd.SigmoidHadamard(z, w)` 把热路径的 `Hadamard(Sigmoid(z), w)` 融合成单节点（采纳于 `nn/ltc.go:347`）。前向逐位为构造性（逐字调用同一批 tensor 算子）；常规 2D 反向通过把 `g⊙w` 乘积在完全相同的位点舍入，达成与旧双节点链的逐位等价；异形或手设种子则逐字回退到旧组合（怪癖与 panic 契约一并保留）。`LTCStep` 2,442 → **2,306** allocs/op（−5.6%）、`UnrollBackward` 33,963 → **31,983**（−5.8%）——个位数收益，且为实测的结构上限：每个点位恰好省一个图节点加一个反向中间张量，剩余由 `tensor.New` 的逐节点固定开销主导（上一项）。见 [architecture.md](architecture.md) |
| CfC 的 `erev` 死梯度 | **已完成（阶段 8）：** CfC 的反转电位现已焙入构造期指示矩阵（indicator matrix）——照搬了 LTC 的手法。`erev`/`sErev` 字段是不带梯度的普通 `*tensor.Tensor`，因此死梯度从"为零"升级为**结构上不可能**；`drive()` 收缩为两次 MatMul。与旧的 `Var` 叶驱动逐位等价（红队差分测试），且加载路径按流内极性重建指示阵（[persistence.md](persistence.md)） |
| 指示矩阵 O(units³) 实体化 | **加载侧已封堵（阶段 8），根因未修：** 稠密的 `[pre·units, units]` 归约指示矩阵使构造期与加载期内存达 O(units³)，而控制它的头部只有 9–13 字节。`LoadLTC`/`LoadCfC` 现把 `units`/`inDim` 封顶在 256（每个加载细胞 ≤ 256 MiB 指示矩阵，峰值约 320 MiB），把一条 13 字节、units=4096 的攻击流——过去会尝试约 550 GB 直到操作系统强杀进程——收敛为一个带值的 error。根因级修复——以稀疏收缩彻底不实体化 `[units², units]` 指示阵（构造器侧同样存在此悬崖）——仍然开放（[persistence.md](persistence.md)） |
| 统一归约形状约定（`SumRows`/`SumCols`、1D 提升） | 单独评估；API 破坏性 |
| `tensor.Stack` | 实验性：产出没有其他算子消费的 3D 张量（`tensor/tensor.go:170`）；为兼容保留 |
| CfC（Closed-form Continuous-time）细胞 | **已完成（阶段 6）：** `nn.CfC`（`nn/cfc.go`）——与 LTC 同一 ODE、同一套突触参数化，以 Lemma 1 闭式解驱动；论文↔代码对照与验证留痕见 [cfc.md](cfc.md)。新 API：仍可能演进 |
| 内置优化器 | **已完成（阶段 6）：** `optimizer` 包（SGD/Momentum/Adam，覆盖率 100%）；手写循环依然是理解引擎以及实现该包未覆盖规则的受支持范式——[training.md](training.md) 覆盖两种形态 |
| 序列化（Save/Load） | **已完成（阶段 7）：** `serialize` 包（带版本的 `"LNNS"` 张量流）外加 `nn` 的六个 Save/Load 函数（LTC/CfC/Linear）；对恶意流安全——只有 error 绝不 panic、固定限额先校验后分配、未知长度读端渐进分配（4 GiB 声明在 18 字节后停止仅峰值约 33 KiB）。红队变异模糊：7,500 个变异体 0 panic，资源耗尽加固后再测 1,200 个变异体，依然 0 panic。格式规格、API 指南与完整安全契约见 [persistence.md](persistence.md) |
| 基准/CI 工具 | **已完成：** `make bench`（13 项基准）+ GitHub Actions CI（gofmt 门禁、vet、build、`test -race`、example 冒烟） |
