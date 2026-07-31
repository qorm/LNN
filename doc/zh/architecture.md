> [English](../architecture.md) | 中文

# 架构

**摘要：** lnn 分为三层——`float32` 数值内核（`tensor`）、动态反向模式自动微分引擎（`autograd`）和模型层（`nn`）——每一层只导入它下面的一层，没有框架、没有代码生成、除标准库外零依赖。

**读者对象：** 需要调试、剖析或扩展本库的工程师。

## 层总览

```
┌──────────────────────────────────────────────────────────────┐
│  你的代码                                                    │
│  数据、模型、损失、更新循环（手写或 optimizer 包）           │
└───────────────┬──────────────────────────────────────────────┘
                │  Step/Forward、Unroll、ParametersOf、
                │  ZeroGrad → Backward → 显式参数更新
┌───────────────▼──────────────────────────────────────────────┐
│  nn — 模型层                                                 │
│  Linear · LTC · Wiring · Cell/Unroll · Module/ParametersOf   │
└───────────────┬──────────────────────────────────────────────┘
                │  把可微算子组合成一张 *autograd.Variable 图
┌───────────────▼──────────────────────────────────────────────┐
│  autograd — 动态图 + 反向模式自动微分                        │
│  Variable{Data, Grad, parents, opKind 标签 + 载荷字段}       │
│  前向：即时执行；每个算子给输出打上算子种类标签              │
│  Backward：逆拓扑序遍历，按标签派发反向传播                  │
└───────────────┬──────────────────────────────────────────────┘
                │  所有前向与反向计算都是普通的张量算子
┌───────────────▼──────────────────────────────────────────────┐
│  tensor — 数值内核                                           │
│  稠密行主序 []float32 · MatMul · 带广播的逐元素运算          │
│  激活 · 归约 · 切片 · 随机数                                 │
└──────────────────────────────────────────────────────────────┘
```

导入方向严格向下（`nn → autograd → tensor`）；没有环，也没有跨层捷径——唯一的例外是 `nn` 会直接调用 `tensor` 来构造不需要梯度的常量（接线掩码（mask）、epsilon、稀疏收缩（sparse contraction）播种用的单位阵与 `+0` 标量、反转电位行视图）。`optimizer` 包（SGD、Momentum、Adam）与 `nn` 并列坐在 `autograd` 之上——它只导入 `autograd`，写的就是手写循环会写的朴素 Go 原地更新（[training.md](training.md)）。`serialize` 包（持久化）与它们并列，坐在 `tensor` 与 `autograd` 之上——它是 `nn` 六个 Save/Load 函数背后的存储层，单独暴露是为了让线上格式（wire format）及其不可信流（untrusted stream）安全契约可以被独立审计（[persistence.md](persistence.md)）。

| 层 | 职责 | 刻意不具备的东西 |
|---|---|---|
| `tensor` | 稠密缓冲与数值内核；以 panic 做校验 | stride、视图、原地算子、枚举子集之外的广播 |
| `autograd` | 即时前向 + 按算子种类（op kind）标签派发的反向传播；反向模式 `Backward` | tape/session 对象、逐节点反向闭包（已被标签取代）、图优化、高阶导数 |
| `nn` | 层、细胞（LTC 与 CfC）、接线、序列展开（unroll）、参数聚合、六个 Save/Load 函数 | 自己的线上格式——持久化是独立的 `serialize` 包；内置优化器位于独立的 `optimizer` 包 |
| `optimizer` | 作用于 `autograd` 叶节点的 SGD/Momentum/Adam，显式 struct | 每参数速度/矩以外的状态；学习率调度（由调用方持有的字段） |
| `serialize` | 带版本的 `"LNNS"` 张量流与模型流；只返回 error 的加载路径，对恶意流安全 | 版本协商（只读 version 1，未知版本直接拒绝）；压缩、加密 |

## 一次训练迭代的数据流

```
ZeroGrad(所有参数)              // 清空叶节点的 Grad 缓冲区
        │
前向：算子即时执行              // 每个算子分配一个新张量，并给它的
        │                       // 输出打上算子种类标签
        ▼
   loss（图的标量叶节点）
        │
loss.Backward()                 // 对图做拓扑排序，逆序派发每个节点
        │                       // 的反向传播；叶节点梯度累加
        ▼
你的更新规则                    // p.Data.Data[i] -= lr * p.Grad.Data[i]
```

完整的循环见根目录 `README.md`（中文版 `README_zh.md`）的快速上手和 [training.md](training.md)；`examples/ltc-sequence` 是端到端的参考实现。

## 为什么张量没有 stride

`Tensor` 恰好就是 `{Shape []int, Data []float32}`。没有 stride 数组，也就没有视图：每一个操作——包括 `SliceRow`、`SliceCol`、`Transpose` 和 `Clone`——都分配一块全新缓冲区并拷贝。

对这个体量的库来说，这个取舍是刻意为之的：

- **别名（aliasing）在构造上不可能。** 除非用户显式共享 `Data` 切片，两个张量永远不会共享存储，因此不存在需要文档化或防御的写后读风险类别，每个反向步骤都可以直接读取本节点的前向张量而无需生命周期记账。
- **已构造对象的不可变性是真实的。** `Wiring` 的掩码只通过拷贝型访问器暴露，红队审计确认：篡改返回的掩码不会影响细胞（指针型访问器就不具备这个性质）。
- **代价是显式且可预期的：** 每个算子 O(元素数) 的拷贝，以及随计算图增长（见下文）的内存。本库面向 CPU 上小而可审计的模型，在这里清晰性胜过吞吐量；一个带 stride 的视图层将是另一个库。

MatMul 确实在内层循环里跳过零乘数，这是唯一一处可观测行为与朴素数学不同的地方——而且该跳过只检查**左**操作数，因此是有方向性的：`0 * NaN` 贡献 `0`（零乘数被跳过），而 `NaN * 0` 仍得 `NaN`（见 [pitfalls.md](pitfalls.md)）。

## 计算图

每个 `autograd.Variable` 都是一个节点：

```go
type Variable struct {
    Data *tensor.Tensor   // 前向值
    Grad *tensor.Tensor   // 累积的梯度（首次反向之前为 nil）

    parents  []*Variable  // 产生该节点的算子的输入
    kind     opKind       // 标签：runBackward 据此派发梯度公式
    scalar   float32      // 载荷：Scale 系数 / Pow 指数
    from, to int          // 载荷：SliceCol 列区间 / SliceRow 行下标
    aux      *tensor.Tensor // 载荷：Div 的分母倒数 / SigmoidHadamard 的 sigmoid 缓冲
    idx      []int        // 载荷：GatherRows 索引（构造时已拷贝）
}
```

- **叶节点**（`Var`、`New`、`Const`）没有 parents，种类为零值（`opLeaf`）——没有反向步骤。它们是参数和输入；梯度传播到它们为止。
- **算子**通过 `tensor` 即时执行前向计算，然后给输出节点打上算子种类（op kind）标签及其载荷——**没有逐节点闭包**。反向步骤曾经是闭包，而闭包捕获会为每个图节点分配一个堆对象：这是深度展开图中最大的分配来源之一。阶段 7 深改以标签派发取而代之：一个 `uint8` 种类加 struct 上的几个载荷字段；`runBackward` 是一个 23 路 switch（每个 `opKind` 一个 case，含空操作的 `opLeaf`），其 case 体恰是原来闭包承载的梯度公式。少数 case（例如 `opLog`）在反向时读取*父节点*的 `Data`，因此叶节点数据在前向和反向之间不得被修改（细节见 [pitfalls.md](pitfalls.md)）。
- **Backward** 用后序 DFS 构建拓扑序，逆序对每个节点执行 `runBackward`，然后把除接收者之外所有非叶节点的 `Grad` 清零。

### 梯度缓冲区是移交的，不是克隆的

`addGrad` 在形状不符时 panic（与从前一样）——而在*首次*贡献时，它直接取得传入缓冲区的所有权（ownership transfer，所有权移交），不做克隆。每个反向 case 传入的要么是全新分配的张量，要么是专门为这次调用准备的缓冲区，因此没有其他引用能观察到后续对它的累加。`Add` 的 case 把这一设计展示得最锋利：a 支经由取得所有权的归约器 `sumToShapeTake` 把 `v.Grad` 本体交给 `a`（安全性由逆拓扑序保证：运行到这一步时 `v` 的所有消费者都已贡献完毕），而 b 支刻意走*克隆版*的 `SumToShape`——当两个操作数都与 `v` 同形时，它必须交给 `b` 一块独立的缓冲区，否则 `a.Grad` 与 `b.Grad` 就会别名（alias），之后向任何一方累加都会腐蚀另一方（设想 `Add(x, y)` 且两个叶都在下游被复用）。根种子在集中处受到同样待遇：`Backward` 从一份私有克隆开始传播，这样自动播种的全 1 缓冲区就可以交给某个叶，而不会别名接收者的 `Grad`——正是这一点让重复 `Backward` 保持严格线性、让手设种子保持完好。逐 case 的所有权审计就在 `autograd/ops.go` 的注释里，别名设计由专项探测钉住（`autograd/alias_probe_test.go`，外加 `autograd/f1_regress_test.go` 的形状契约回归）——反向分配中 Clone 的占比从约 20% 降到约 1%。

`sumToShapeTake` 本身自 v0.4.0 起是 autograd 内部实现：它曾是导出的 `tensor.SumToShapeTake`，但它唯一的调用方就是 `autograd/ops.go` 里那五处反向点位，把所有权脚枪（footgun）挂在公共 tensor 面上换不来任何外部价值——于是它内移并改为非导出（语义、所有权契约与 panic 文案逐字保持）。克隆版、无别名的 `SumToShape` 仍是公共面上唯一的归约器。

### 融合反向——以及 FMA 屏障

同一次深改把多节点梯度链融合（fused backward，融合反向）成单循环：一元激活（Sigmoid、Tanh 从 4 个图节点 → 1 个；Log、Pow 从 3 → 1；……）、MatMul 反向改经新的转置感知内核 `MatMulTransA`/`MatMulTransB`（乘积与累加次序与显式 `Transpose` 之上的 MatMul 逐字相同，只省掉两块转置缓冲）、以及 `hadamardReduce` 对 Hadamard 与 Div 操作数的「乘积或归约后乘积」融合。

其中值得讲给读者的工程决策是 **FMA 屏障**。朴素的融合乘加循环——`r[j] += g[j] * x[j]`——会被 arm64 的 SSA 后端编译成 `FMADD` 指令：一次舍入；而历史上的两步路径舍入两次（乘积先存进张量元素，再累加）。漂移是每元素约 1 ULP——单个算子层面不可见，但作为十万节点图上的逐位等价宣称则是灾难性的。屏障就是 `mul32`：

```go
func mul32(a, b float32) float32 {
    if math.Float32bits(a)&0x7F800000 == 0x7F800000 ||
        math.Float32bits(b)&0x7F800000 == 0x7F800000 {
        return a * b
    }
    return float32(float64(a) * float64(b))
}
```

对有限操作数，精确乘积装得进 48 位尾数，`float64` 能精确表示，因此舍入回 `float32` 与硬件乘法逐位相同——而这个转换在 SSA 图中留下一个显式舍入节点，FMA 成形规则无法跨越它。非有限操作数（NaN 或 ±Inf）改走**原生** `a * b` 路径：float64 往返会把 NaN 载荷重新规范化（recanonicalize），偏离旧链的硬件 float32 传播行为。原生乘积在其表达式中是孤立的乘法、邻接不到加减，且该分支在 SSA 图中留下 FMA 规则匹配不到的 If/Phi 结构，因此融合同样够不着它——以 `go tool compile -S` 验证：全包 `FMADD` 族指令为零，而裸循环对照探针仍然发射它们（屏障是承重的，不是拜物仪式）。相关的细节：负号融合乘以包级*变量* `negOne`（存 −1），因为编译器会把乘以常量 −1 折叠成符号翻转（`FNEGS`），那会翻转 NaN 的符号位，而旧链的硬件乘法不会。

处处是忠实优先于修正：历史上 1D→`[1, n]` 的叶梯度形状怪癖被忠实复刻（`elemwiseGradShape`）；异形的手设梯度形状回退到字面的旧组合（`gradMatchesElemwise` 守卫）——panic 契约一并保留。整个深改以对 git 历史中提取的重写前实现做差分模糊测试为门禁：52,000 张图，四类差异（有限值、形状、panic 有无、NaN 位）全零，负对照确认门禁确实能侦测注入的变异。本机实测（`-benchtime=100x`）：`UnrollBackward` 68,688 → **33,963 allocs/op（−50.55%）**，字节数 −50.1%、耗时 −24%——另外四项基准全降、零回归：`ChainForwardBackward` −57.7%、`DivDenLoop` −56.7%、`LTCStep` −29.0%、`GatherRowsBackward` −23.5%。

### Sigmoid–Hadamard 融合：阶段 8 的收尾

阶段 8 的 `autograd.SigmoidHadamard(z, w)`（`autograd/ops.go`）把 LTC 热路径模式 `Hadamard(Sigmoid(z), w)`——两个图节点——融合为一个，采纳于 `synapsesRows` 感知/循环共用的唯一入口（`nn/ltc.go:423`）。它是反向深改的收尾之作，因为它是唯一一个需要*新算子*而非既有算子重组的融合，其等价性叙事有三段彼此不同：

- **前向逐位为构造性。** 它逐字调用旧组合所跑的同一批 tensor 算子——先 `tensor.Sigmoid`，再 `tensor.Hadamard`——因此形状、广播与数值按定义相同，而非靠测量得出。sigmoid 缓冲被保存在节点的 `aux` 槽里，反向直接复用而不重算。
- **常规反向以舍入位点对齐达成逐位。** 在 2D 路径（LTC 热路径）上，反向以一个融合循环传播 `dz = g⊙w⊙s⊙(1−s)`，并以 Hadamard 反向所用的同一个 `hadamardReduce` 传播 `dw = g⊙s`。这*并非*设计成逐位的——设计书预期的是一个容差门禁——但结果却是逐位的：把 `g⊙w` 乘积恰在旧 Hadamard 反向把中间张量交给 sigmoid 节点的那个位点经 `mul32` 舍入，便复刻了旧中间张量的舍入；外层 `mul32(gw, s⊙(1−s))` 的分组则复刻了 opSigmoid 的融合循环。结果在常规路径上与旧双节点链逐位相同（由测试钉住），这也正是两个 example 的训练轨迹一位都不曾漂移的原因。
- **回退路径是逐字复刻。** 非 2D 操作数或异形手设梯度会逐字复刻旧的 `opHadamard`+`opSigmoid` 对——两条 `hadamardReduce` 分支，再接 opSigmoid 自己的派发——因此数值、形状、1D→`[1,n]` 升维怪癖与 panic 契约全部原样保留。

同一 A/B 时间窗实测（`-benchtime=100x`）：`LTCStep` 2,442 → **2,306 allocs/op（−5.6%）**、`UnrollBackward` 33,963 → **31,983（−5.8%）**。这个收益被刻意如实报告为它实际所在的个位数：每个融合点位恰好省一个图节点加一个反向中间张量，且结构核算精确闭合（`LTCStep` 68 点位 × 2 次分配 = 136 = 差值；`UnrollBackward` 396 点位 × 5 = 1,980）。剩下的就是 `tensor.New` 的逐节点固定开销（下一节处理）——这是本引擎上融合所能触及的实测结构上限，也正是早先那个"收益/脆弱性"之问一直在等的答案。

### 内嵌形状 backing——阶段 10 的分配削减

自阶段 6 起作为路线图条目 #12 追踪的逐节点固定开销：每个图节点都要为它的前向输出张量付费，而每次张量构造过去付两次——一次是带 `Data` 缓冲的 `Tensor` 本体，一次是堆上的 `Shape` `[]int` 切片。剖析显示这项占剩余分配的 64.9%（实施前复测 60.4%）。v0.4.0 以**内嵌 `[4]int` 形状缓冲**（embedded backing，结构体内联的 `shapeBuf`）消除了 `Shape` 这一半：`Shape` 仍是 `[]int` 的唯一真相源，但 rank ≤ 4 时它指向结构体自身，零堆分配；超过 4 的秩——只有 `serialize` 的线上流会承载至多 rank 8 的张量——回退到拷贝的堆切片，兼容性得以保留。

这条决策痕迹值得留存，因为诚实的数字同时反对*又*支持这一改动。原型实测五基准 **allocs −18~−26%**（确定性；tensor 八项算子基准每算子恰少一次 shape 分配，计数 4→3 处即 −25%），但交替 A/B 下墙钟落在 **±数% 噪声内**、`B/op` 上升约 3%（内嵌缓冲使每个 `Tensor` 变大）：收益在分配次数与 GC 卫生，而非吞吐。台面上有两种实现：选①值类型形状字段能消除同一次分配，却要破坏全部 **233** 处 `.Shape` 访问加 **7** 处直写——收益相同的 API 断裂；选②内嵌 backing 只触及约 10 个内部点位，零破坏（读路径不动、导出字段类型不变）。库主拍板选②；唯一的 API 新增是导出的 `Tensor.Reshape`——`t.Shape = …` 直写的受权替代品（负维度 panic）。同一轮 v0.4.0 API 卫生改动还删除了 `tensor.Stack`（3D 产出无算子消费、库内零调用），并把所有权契约的 `SumToShapeTake` 移入 autograd 内部（见上）。

### 非叶节点的梯度在每次遍历后被置 nil——以及为什么

中间梯度在设计上是瞬态的。如果过期的中间 `Grad` 缓冲区在一次遍历后存活下来，对同一张图的第二次 `Backward` 就会穿过已被播种的节点继续传播，叶节点梯度将超线性增长（红队在修复前的代码上实测两次调用得到 3 倍）。清零之后，重复运行变成严格线性：对同一张图调用 N 次，得到单次的 N 倍叶梯度。受支持的范式仍然是每张新图一次 `Backward`；线性的重复运行行为是一个定义好的安全网，而不是用来依赖的特性（[pitfalls.md](pitfalls.md)）。

### 图就是内存模型

每个中间张量都保持存活——被其节点引用——直到 `Backward` 完成。因此内存随每次迭代的算子数量扩展，而不仅仅随参数规模。一次 LTC `Step` 会把 `unfolds` 轮 ODE 迭代展开进图；自阶段 6 的突触向量化起，每轮是 O(units) 个激活块；自阶段 9 的稀疏收缩（sparse contraction）起，突触前轴归约是 `+0` 播种、末端归一化 MatMul 收尾的折叠（fold）（见 [ltc.md](ltc.md)）——从 O(units²) 个逐突触节点降下来，也告别了阶段 9 所消灭的稠密 `[units², units]` 指示矩阵（indicator matrix）（那会在构造期*与*加载期实体化 O(units³) 个 float32；见 [persistence.md](persistence.md)）。阶段 7 的反向深改（见上）把逐节点分配数砍掉一半，阶段 8 的 Sigmoid–Hadamard 融合再进一步，阶段 10 的内嵌形状 backing（embedded backing，见上）把每次张量构造再削去一次分配。当前实测：`LTCStep` **2,707 allocs/op**、`UnrollBackward` **31,994**——较最初的逐突触循环（7,360 / 120,163）累计 **−63% 与 −73%**。阶段 9 这一步是诚实的权衡：allocs 相对阶段 8 数值（2,306 / 31,983）*上升*约 43%/30%（折叠每级的克隆），但 ns/op *下降*约 21%/13%（红队独立复测），因为稠密指示阵 MatMul 对零行的 O(units³) 空转内层循环与反向大分配消失了——**分配次数换走了无用算力，墙钟净受益**。同一改动还消除了*构造期*内存悬崖：`units = 1024` 全接线细胞现在分配约 32 MB（实测：`NewLTC` 36.4 MiB、`NewCfC` 32.4 MiB；红队复测一致），而不再是旧的约 8 GiB 指示阵。图内存仍然随 `units · unfolds · 序列长度` 增长，所以在这个引擎上三者都要保持适度；`CfC` 细胞（[cfc.md](cfc.md)）的闭式步进则完全没有 `unfolds` 因子。

## float32 是全局约束

所有存储和所有公开 API 都是 `float32`；没有 `float64` 模式，也没有混合精度方案。需要余量的内核会在内部提升到 `float64` 再舍入回来——`Sigmoid`、`Tanh`、`Exp`、`Log`、`Pow`、`Softplus` 以 `float64` 求值 `math.*`；`LogSoftmaxRows` 以 `float64` 累加其归一化和；`Randn` 以 `float64` 跑 Box-Muller；`LTC.scaledCapacitance` 在转换回 `float32` 之前以 `float64` 计算并钳制 `cm/dt` 缩放（见 [ltc.md](ltc.md)）。需要据此规划的后果：

- 约 7 位十进制有效数字；累加误差随归约长度增长。
- `Exp` 作用于约 88 以上的值会溢出为 `+Inf`；没有全局防护（softmax 族在内部做了稳定化，其余都是普通算术）。
- 没有对非规格化数友好的内核；非规格数量级会直接冲刷到零。

## 并发

设计上单线程——`Backward` 在无同步的情况下修改叶节点的 `Grad` 缓冲区，且张量直接暴露存储。受支持的并行模式是按 goroutine 隔离的实例（各自的细胞、张量、RNG、计算图），绝不共享状态。契约和一个已验证的模式见 [pitfalls.md](pitfalls.md)。
