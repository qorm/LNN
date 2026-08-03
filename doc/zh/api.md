# API 速查

> [English](../api.md) | 中文

**本页定位：** 按包组织的单页导航层——每个导出符号一行，附 panic/error
画像与指向权威文档的锚点。**它不是 godoc 的复制品：** 每个符号的权威契约
（完整参数语义、panic 条件、错误分类、位级说明）在 godoc 里，签名背后的
概念在页尾链接的指南里。

## 如何使用本 API

godoc 即契约。三种入口：

```sh
go doc github.com/qorm/LNN/nn.NewLTC      # 单个符号
go doc github.com/qorm/LNN/tensor          # 整个包（panic 约定、布局、广播）
go doc -all github.com/qorm/LNN/autograd   # 包内全部符号
```

在线：[pkg.go.dev/github.com/qorm/LNN](https://pkg.go.dev/github.com/qorm/LNN)
（[tensor](https://pkg.go.dev/github.com/qorm/LNN/tensor) ·
[autograd](https://pkg.go.dev/github.com/qorm/LNN/autograd) ·
[nn](https://pkg.go.dev/github.com/qorm/LNN/nn) ·
[optimizer](https://pkg.go.dev/github.com/qorm/LNN/optimizer) ·
[serialize](https://pkg.go.dev/github.com/qorm/LNN/serialize)）。

**标记列：** `panic` = 误用或非法输入以 panic 报告（全库默认纪律——输入来自
你自己的程序，坏形状就是调用方的 bug）；`error` = 失败以 error 返回
（持久化路径专有，它们消费不可信的字节流——那里绝不 panic）；`—` = 除显然
情形（如 nil 接收者解引用）外无失败模式。

两条贯穿全表的契约：

- **形状** —— `[m, n]` 表示 2D 行主序张量；广播规则是显式枚举的子集，归约
  输出形状不对称。依赖任何输出形状之前，先读
  [shapes-and-broadcasting.md](shapes-and-broadcasting.md)。
- **梯度累加** —— 叶节点的 `Grad` 跨 `Backward` 调用累加；每次反向之前对
  每个参数 `ZeroGrad`（[training.md](training.md)）。

## tensor — [godoc](https://pkg.go.dev/github.com/qorm/LNN/tensor) · [形状指南](shapes-and-broadcasting.md)

稠密行主序 `float32` 张量。`Tensor` 就是 `Shape []int` + 扁平
`Data []float32`；每个算子都分配全新缓冲区（无视图、无别名，除非你故意
共享 `Data`）。

### 构造

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`Tensor`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor) | `struct{ Shape []int; Data []float32 }` | 稠密行主序张量；`[m,n]` 的元素 `(i,j)` 在 `Data[i*n+j]` | — |
| [`New`](https://pkg.go.dev/github.com/qorm/LNN/tensor#New) | `New(shape ...int) *Tensor` | 零填充；`New()` 是秩 0、含一个零 | panic：负维度、int64 溢出 |
| [`FromData`](https://pkg.go.dev/github.com/qorm/LNN/tensor#FromData) | `FromData(data []float32, shape ...int) *Tensor` | 复制 `data`（不共享） | panic：size≠len(data)、负维度、溢出 |
| [`FromRows`](https://pkg.go.dev/github.com/qorm/LNN/tensor#FromRows) | `FromRows(rows ...[]float32) *Tensor` | `[len(rows), len(rows[0])]`，逐行复制；空输入 → `[0,0]` | panic：行长度不一 |
| [`Reshape`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Reshape) | `(*Tensor).Reshape(dims ...int)` | 重指 `Shape` 而不动 `Data`；秩 > 4 回退到堆上形状切片 | panic：负维度；**不**校验元素总数 |
| [`Clone`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Clone) | `(*Tensor).Clone() *Tensor` | 深拷贝，不共享存储 | — |
| [`ZerosLike`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.ZerosLike) / [`OnesLike`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.OnesLike) | `(*Tensor) … () *Tensor` | 同形状的零/一张量 | — |
| [`SameShape`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SameShape) | `SameShape(a, b *Tensor) bool` | 维度列表全等 | — |

### 线性代数（仅 2D）

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`MatMul`](https://pkg.go.dev/github.com/qorm/LNN/tensor#MatMul) | `MatMul(a, b) *Tensor` | `[m,k] × [k,n] → [m,n]`；仅限矩阵乘法 | panic：非 2D、内维不等 |
| [`MatMulTransA`](https://pkg.go.dev/github.com/qorm/LNN/tensor#MatMulTransA) | `MatMulTransA(a, b) *Tensor` | `aᵀ·b`，`[m,k] 与 [m,n] → [k,n]`，不分配转置缓冲 | panic：非 2D、行数不等 |
| [`MatMulTransB`](https://pkg.go.dev/github.com/qorm/LNN/tensor#MatMulTransB) | `MatMulTransB(a, b) *Tensor` | `a·bᵀ`，`[m,k] 与 [n,k] → [m,n]`，不分配转置缓冲 | panic：非 2D、列数不等 |
| [`Transpose`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Transpose) | `Transpose(a) *Tensor` | `[m,n] → [n,m]` | panic：非 2D |

### 逐元素（任意秩；二元算子 = 枚举广播）

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`Add`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Add) / [`Sub`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Sub) / [`Hadamard`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Hadamard) | `(a, b *Tensor) *Tensor` | 在枚举广播规则下的 `+`、`−`、逐元素 `*`；1D⊕1D → `[1,n]`，`[1]⊕[1]` → `[1,1]`，`[m,1]⊗[n]` → 外积 `[m,n]` | panic：不可广播 |
| [`Scale`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Scale) | `Scale(a, s float32) *Tensor` | 每元素 × s | — |
| [`Neg`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Neg) | `Neg(a) *Tensor` | `Scale(a, -1)` | — |
| [`Apply`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Apply) | `Apply(a, f func(float32) float32) *Tensor` | 按行主序扁平次序映射 `f` | — |
| [`Tanh`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tanh) / [`Sigmoid`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Sigmoid) / [`Exp`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Exp) / [`Log`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Log) / [`Pow`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Pow) / [`Softplus`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Softplus) | 一元：`(a) *Tensor`；`Pow(a, p)` | 标准激活/数学函数，sigmoid 与 softplus 数值稳定；不查定义域——`Log`/`Pow` 按 float32 语义产生 NaN/Inf | — |
| [`Clip`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Clip) | `Clip(a, lo, hi float32) *Tensor` | 钳位到 `[lo, hi]`；约定 `lo ≤ hi` | — |

### 归约

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`SumAll`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SumAll) | `SumAll(a) *Tensor` | 总和 → 形状 `[1]`；空张量 → 0 | — |
| [`MeanAll`](https://pkg.go.dev/github.com/qorm/LNN/tensor#MeanAll) | `MeanAll(a) *Tensor` | 均值 → 形状 `[1]` | panic：空张量 |
| [`SumRows`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SumRows) | `SumRows(a) *Tensor` | `[m,n]` 沿轴 0 求和 → **`[1,n]`**（保持 2D） | panic：非 2D |
| [`SumCols`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SumCols) | `SumCols(a) *Tensor` | `[m,n]` 沿轴 1 求和 → **`[m]`**（1D）——与 SumRows 不对称，冻结约定 | panic：非 2D |
| [`SumToShape`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SumToShape) | `SumToShape(grad, shape) *Tensor` | 反向归约器：同形→克隆，标量→`[1]` 总和，`[n]`/`[1,n]`→列和，`[m,1]`→行和（完整表见指南） | panic：其余任何目标形状 |
| [`SoftmaxRows`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SoftmaxRows) / [`LogSoftmaxRows`](https://pkg.go.dev/github.com/qorm/LNN/tensor#LogSoftmaxRows) | `(a) *Tensor` | 逐行（log-）softmax，减最大值形式数值稳定；零列 → 空 `[m,0]` | panic：非 2D |

### 切片与访问

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`SliceCol`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SliceCol) | `SliceCol(a, from, to int) *Tensor` | 列 `[from,to)` → `[m, to-from]` 副本 | panic：非 2D、区间非法或为空 |
| [`SliceRow`](https://pkg.go.dev/github.com/qorm/LNN/tensor#SliceRow) | `SliceRow(a, i int) *Tensor` | 第 `i` 行 → `[1,n]` 副本 | panic：非 2D、`i` 越界 |
| [`ConcatCol`](https://pkg.go.dev/github.com/qorm/LNN/tensor#ConcatCol) | `ConcatCol(ts ...*Tensor) *Tensor` | `[m,n1],[m,n2],… → [m, Σn]` | panic：零输入、非 2D、行数不一 |
| [`At`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.At) / [`Set`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Set) | `At(idx ...int) float32` / `Set(v float32, idx ...int)` | 按索引取/置单个元素，任意秩 | panic：索引个数不符、越界 |
| [`Rows`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Rows) / [`Cols`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Cols) | `(*Tensor) … () int` | 第一/第二维 | panic：非 2D |
| [`Dims`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Dims) / [`Size`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Size) | `(*Tensor) … () int` | 秩 / 元素总数 | panic（Size）：int64 溢出 |
| [`Scalar`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.Scalar) / [`IsScalar`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.IsScalar) / [`IsRowVec`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.IsRowVec) | 见 godoc | 单元素判定/提取；行向量判定（`[n]` 或 `[1,n]`） | panic（Scalar）：非单元素 |
| [`String`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Tensor.String) | `(*Tensor).String() string` | 调试渲染；超 64 个元素只显示首尾 | — |

### 随机

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`Uniform`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Uniform) | `Uniform(rng, lo, hi float32, shape ...int) *Tensor` | U(lo, hi)；`lo > hi` 时区间镜像（遗留行为） | panic：nil rng、负维度 |
| [`Randn`](https://pkg.go.dev/github.com/qorm/LNN/tensor#Randn) | `Randn(rng, shape ...int) *Tensor` | N(0,1) Box-Muller；尾部在 ≈ 7.43σ 处截断（已文档化） | panic：nil rng、负维度 |

## autograd — [godoc](https://pkg.go.dev/github.com/qorm/LNN/autograd) · [训练](training.md) · [架构](architecture.md)

反向模式自动微分的动态计算图。每个算子即时求值并在输出节点打上操作标记；
`Backward` 按逆拓扑序遍历整图。`Backward` 运行之前整图常驻内存。

### 图与叶节点

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`Variable`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Variable) | `struct{ Data, Grad *tensor.Tensor }` | 图节点：值 + 累积梯度 | — |
| [`Var`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Var) | `Var(t) *Variable` | 叶节点；**别名** `t`（不复制）——原地更新直接生效 | — |
| [`New`](https://pkg.go.dev/github.com/qorm/LNN/autograd#New) | `New(data []float32, shape ...int) *Variable` | 叶节点；复制 data（经 `tensor.FromData`） | panic：形状/长度不符 |
| [`Const`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Const) | `Const(t) *Variable` | `Var` 的别名，标明常量意图；梯度仍会流入——忽略即可 | — |
| [`Detach`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Detach) | `Detach(v) *Variable` | 以 `v` 的值新建无父叶节点——**别名** `v.Data`（零拷贝）；切断流向 `v` 祖先的梯度，但不断存储（原地参数更新仍会带动被 detach 的参数） | — |

### 算子（均返回新节点；前向 panic = 所包装 `tensor` 算子的 panic）

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`MatMul`](https://pkg.go.dev/github.com/qorm/LNN/autograd#MatMul) | `(a, b) *Variable` | `[m,k]×[k,n]→[m,n]` | panic：非 2D、内维不等 |
| [`Add`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Add) / [`Sub`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Sub) / [`Hadamard`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Hadamard) | `(a, b) *Variable` | 广播算子；反向经 `SumToShape` 归约到各操作数形状 | panic：不可广播 |
| [`Scale`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Scale) / [`Neg`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Neg) | `Scale(a, s)` / `Neg(a)` | 常数缩放 / 取负 | — |
| [`Tanh`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Tanh) / [`Sigmoid`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Sigmoid) / [`Exp`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Exp) / [`Log`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Log) / [`Pow`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Pow) / [`Softplus`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Softplus) / [`Abs`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Abs) / [`Relu`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Relu) | 一元 `(a) *Variable`；`Pow(a, p)` | 反向均为融合实现；`Pow(_, 0)` 梯度恰为 0；`Abs`/`Relu` 在 0 处取梯度 0 | — |
| [`Div`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Div) | `(a, b) *Variable` | 商法则；`b==0` 前向得 ±Inf（float32 除法语义） | — |
| [`SigmoidHadamard`](https://pkg.go.dev/github.com/qorm/LNN/autograd#SigmoidHadamard) | `(z, w) *Variable` | 融合 `Sigmoid(z)⊙w`（LTC/CfC 热点路径），单节点 + 复用 sigmoid 缓冲 | panic：不可广播 |
| [`ConcatCol`](https://pkg.go.dev/github.com/qorm/LNN/autograd#ConcatCol) | `(vs ...*Variable) *Variable` | `[m, Σn]`；反向把梯度切回各输入 | panic：零输入、非 2D、行数不一 |
| [`SliceCol`](https://pkg.go.dev/github.com/qorm/LNN/autograd#SliceCol) / [`Col`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Col) | `SliceCol(a, from, to)` / `Col(a, i)` | `[m, to-from]` / `[m, 1]`；反向零填充回写 | panic：非 2D、区间非法 |
| [`SliceRow`](https://pkg.go.dev/github.com/qorm/LNN/autograd#SliceRow) | `(a, i) *Variable` | `[1, n]`；反向零填充回写 | panic：非 2D、`i` 越界 |
| [`SumAll`](https://pkg.go.dev/github.com/qorm/LNN/autograd#SumAll) / [`MeanAll`](https://pkg.go.dev/github.com/qorm/LNN/autograd#MeanAll) | `(a) *Variable` | 标量 `[1]`；反向广播 `g` / `g/size` | panic（MeanAll）：空张量 |
| [`GatherRows`](https://pkg.go.dev/github.com/qorm/LNN/autograd#GatherRows) | `(a, idx []int) *Variable` | `out[i] = a[i, idx[i]]` → 1D `[rows]`；idx 入场即复制 | panic：非 2D、len(idx)≠行数、idx 越界 |
| [`LogSoftmaxRows`](https://pkg.go.dev/github.com/qorm/LNN/autograd#LogSoftmaxRows) | `(a) *Variable` | 稳定的逐行 log-softmax，融合反向 | panic：非 2D |
| [`FusedOp`](https://pkg.go.dev/github.com/qorm/LNN/autograd#FusedOp) | `FusedOp(data, parents, backward) *Variable` | 自定义算子节点：前向由调用方算好，闭包负责该节点**全部**反向工作——对父节点的每一次 `addGrad` 以及被替换子图的累加序都是闭包自己的契约（融合 LTC 的集成点，`nn/ltc_fused.go`） | panic：nil backward |

### 反向与检视

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`Backward`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Variable.Backward) | `(*Variable).Backward()` | 从标量接收者反向传播；叶梯度跨调用、跨图**累加**；非叶节点梯度遍历后清空 | panic：接收者非标量且未预置 `Grad` |
| [`ZeroGrad`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Variable.ZeroGrad) | `(*Variable).ZeroGrad()` | `Grad = nil`——每次反向之前必须调用 | — |
| [`Value`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Variable.Value) | `(*Variable).Value() float32` | 读取单元素节点（损失值） | panic：非单元素 |
| [`TopoOrder`](https://pkg.go.dev/github.com/qorm/LNN/autograd#TopoOrder) | `TopoOrder(v) []*Variable` | 以 `v` 为根的图的 DFS 后序（父先于子，按构造次序）——与 `Backward` 的构建序完全一致；只读内省 | — |
| [`Parents`](https://pkg.go.dev/github.com/qorm/LNN/autograd#Variable.Parents) | `(*Variable).Parents() []*Variable` | 节点的父列表，按构造次序（叶节点为空）；**每次调用返回新副本**——修改它不会重接图 | — |

## nn — [godoc](https://pkg.go.dev/github.com/qorm/LNN/nn) · [ltc](ltc.md) · [cfc](cfc.md) · [持久化](persistence.md)

### 模块

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`Module`](https://pkg.go.dev/github.com/qorm/LNN/nn#Module) | `interface{ Parameters() []*autograd.Variable }` | 一切拥有可训练参数者 | — |
| [`ParametersOf`](https://pkg.go.dev/github.com/qorm/LNN/nn#ParametersOf) | `ParametersOf(modules ...Module) []*autograd.Variable` | 扁平化；**顺序即位置契约**（持久化 API 按位置读写） | — |
| [`Linear`](https://pkg.go.dev/github.com/qorm/LNN/nn#Linear) | `struct{ W, B *autograd.Variable }` | `y = x @ W + b`；`W [in,out]`、`B [out]` | — |
| [`NewLinear`](https://pkg.go.dev/github.com/qorm/LNN/nn#NewLinear) | `NewLinear(in, out int, rng) *Linear` | Xavier 均匀分布 `W`、零偏置 `B` | panic：nil rng、负维度 |
| [`Forward`](https://pkg.go.dev/github.com/qorm/LNN/nn#Linear.Forward) | `(*Linear).Forward(x) *autograd.Variable` | `[batch,in] → [batch,out]` | panic：`x` 非 2D、宽度不符 |
| [`Parameters`](https://pkg.go.dev/github.com/qorm/LNN/nn#Linear.Parameters) | `(*Linear).Parameters() []*autograd.Variable` | 固定顺序：`W`、`B` | — |

### 细胞与序列

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`Cell`](https://pkg.go.dev/github.com/qorm/LNN/nn#Cell) | `interface{ Step(x, h, ts) (out, hNew); StateSize() int }` | 单步 RNN 推进；`x [batch,inDim]`、`h [batch,units]` 或 nil | — |
| [`Unroll`](https://pkg.go.dev/github.com/qorm/LNN/nn#Unroll) | `Unroll(cell, xs, h0, ts) (ys, hN)` | 在一张图里把细胞展开过整个序列——一次 `Backward` 贯穿时间；空 `xs` → 空 `ys`、`hN = h0` | panic：随 `Step` |
| [`UnrollRemat`](https://pkg.go.dev/github.com/qorm/LNN/nn#UnrollRemat) | `UnrollRemat(cell, params, xs, h0, ts, chunkSize, lossFn) (ys, hN, loss)` | 重实体化（rematerialization，梯度检查点）分块 BPTT：梯度与 `Unroll` + `lossFn` + `loss.Backward()` **逐位相等**，峰值图内存由 O(len(xs)) 降为 O(chunkSize)——对抗性 loss 访问序的最坏情形可超过全展开（已如实文档化）；`params` 必须列出**每一个** `Step` 消费的可训练叶（完备性有审计），细胞逐步图结构须与值无关；返回的 `ys`/`hN` 是 detach 的（可安全读取，背后无图） | panic：chunkSize < 1、params 审计、loss 侧消费者次序、跨类共享叶、随 `Step` |
| [`LTC`](https://pkg.go.dev/github.com/qorm/LNN/nn#LTC) | struct | 液态时间常数细胞（Hasani 2021）：`unfolds` 个子步的半隐式 Euler，softplus 正性约束，固定 ±1 反转电位（不在 `Parameters()` 中） | — |
| [`NewLTC`](https://pkg.go.dev/github.com/qorm/LNN/nn#NewLTC) | `NewLTC(inDim, units, wiring, unfolds, rng) *LTC` | nil wiring = 全连接；初始化区间遵循参考实现；固定种子 → 位级一致 | panic：维度 < 1、unfolds < 1、掩码形状不符、nil rng |
| [`CfC`](https://pkg.go.dev/github.com/qorm/LNN/nn#CfC) | struct | 闭式连续时间细胞（Hasani 2022）：同一膜 ODE 以 Lemma 1 闭式解推进，无需展开 | — |
| [`NewCfC`](https://pkg.go.dev/github.com/qorm/LNN/nn#NewCfC) | `NewCfC(inDim, units, wiring, rng) *CfC` | 与 LTC 同参数化，少 `unfolds` | panic：维度 < 1、掩码形状不符、nil rng |
| [`Step`](https://pkg.go.dev/github.com/qorm/LNN/nn#LTC.Step) | `(*LTC/*CfC).Step(x, h, ts) (out, hNew)` | `out [batch,units]`（仿射映射）、`hNew [batch,units]` 原始状态；**ts 必须为正且有限** | panic：NaN/±Inf/≤0 的 ts、x/h 形状错误 |
| [`StateSize`](https://pkg.go.dev/github.com/qorm/LNN/nn#LTC.StateSize) | `(*LTC/*CfC).StateSize() int` | 即 `units` | — |
| [`Parameters`](https://pkg.go.dev/github.com/qorm/LNN/nn#LTC.Parameters) | `(*LTC/*CfC).Parameters() []*autograd.Variable` | 13 个变量，**顺序冻结**（Save* 流内顺序、optimizer 状态的位置键） | — |

### 接线

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`Wiring`](https://pkg.go.dev/github.com/qorm/LNN/nn#Wiring) | struct | 二值突触掩码；构造后不可变，只以副本外露 | — |
| [`FullyConnected`](https://pkg.go.dev/github.com/qorm/LNN/nn#FullyConnected) | `FullyConnected(inDim, units) *Wiring` | 全部突触存在 | panic：维度 < 1 |
| [`RandomSparse`](https://pkg.go.dev/github.com/qorm/LNN/nn#RandomSparse) | `RandomSparse(inDim, units, sensoryP, recurrentP, rng) *Wiring` | 每条突触以概率 p 独立存在 | panic：维度 < 1、p ∉ [0,1]（拒绝 NaN）、nil rng |
| [`Sensory`](https://pkg.go.dev/github.com/qorm/LNN/nn#Wiring.Sensory) / [`Recurrent`](https://pkg.go.dev/github.com/qorm/LNN/nn#Wiring.Recurrent) | `(*Wiring) … () *tensor.Tensor` | 掩码副本：`[inDim, units]` / `[units, units]` | — |
| [`SensoryRow`](https://pkg.go.dev/github.com/qorm/LNN/nn#Wiring.SensoryRow) / [`RecurrentRow`](https://pkg.go.dev/github.com/qorm/LNN/nn#Wiring.RecurrentRow) | `(*Wiring) … (i int) *tensor.Tensor` | 第 `i` 行，`[1, units]` 副本 | panic：`i` 越界 |

### 持久化（只报 error，绝不 panic——见 [persistence.md](persistence.md)）

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`SaveLTC`](https://pkg.go.dev/github.com/qorm/LNN/nn#SaveLTC) / [`LoadLTC`](https://pkg.go.dev/github.com/qorm/LNN/nn#LoadLTC) | `(w, *LTC) error` / `(r) (*LTC, error)` | kind 0 + 头 `inDim, units, unfolds` + 17 个张量；加载位级精确、与种子无关 | error：I/O；加载：kind 不符、截断（`io.ErrUnexpectedEOF`）、unfolds > 1024、units/inDim > 2048、掩码/反转电位非法、版本偏移 |
| [`SaveCfC`](https://pkg.go.dev/github.com/qorm/LNN/nn#SaveCfC) / [`LoadCfC`](https://pkg.go.dev/github.com/qorm/LNN/nn#LoadCfC) | `(w, *CfC) error` / `(r) (*CfC, error)` | kind 1 + 头 `inDim, units` + 17 个张量 | error：同 LoadLTC 去掉 unfolds 项 |
| [`SaveLinear`](https://pkg.go.dev/github.com/qorm/LNN/nn#SaveLinear) / [`LoadLinear`](https://pkg.go.dev/github.com/qorm/LNN/nn#LoadLinear) | `(w, *Linear) error` / `(r) (*Linear, error)` | kind 2 + `W`、`B`；维度藏在 `W` 的形状里 | error：I/O；加载：kind 不符、张量数 ≠ 2、`W` 非 2D、偏置形状不符 |

## optimizer — [godoc](https://pkg.go.dev/github.com/qorm/LNN/optimizer) · [训练](training.md) · [持久化](persistence.md)

显式更新规则，恰好打包手写循环：`ZeroGrad` → 前向 → `Backward` →
`Step(params)`。`Step` 从不清零梯度，并跳过 `Grad` 为 nil 的参数。
超参数是可热替换的导出字段。

### 优化器

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`Optimizer`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#Optimizer) | `interface{ Step(params []*autograd.Variable) }` | 原地更新契约 | — |
| [`SGD`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#SGD) / [`NewSGD`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewSGD) | `struct{ LR float32 }` / `NewSGD(lr) *SGD` | `p -= LR·g`，无状态 | panic：lr ≤ 0 或 NaN（+Inf 被接受，产生 Inf 更新） |
| [`Momentum`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#Momentum) / [`NewMomentum`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewMomentum) | `struct{ LR, Mu float32 }` / `NewMomentum(lr, mu)` | 重球法：`v = Mu·v + g; p -= LR·v`；速度按参数指针键控 | panic：lr ≤ 0、mu ∉ [0,1)；Step：参数尺寸变更 |
| [`Adam`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#Adam) / [`NewAdam`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewAdam) | `struct{ LR, Beta1, Beta2, Eps float32 }` / `NewAdam(lr, b1, b2, eps)` | 偏差校正动量，全程 float32；状态按参数指针键控 | panic：lr ≤ 0、beta ∉ [0,1)、eps ≤ 0；Step：参数尺寸变更 |
| [`NewAdamDefault`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewAdamDefault) | `NewAdamDefault(lr) *Adam` | Kingma & Ba 推荐值：0.9 / 0.999 / 1e-8 | panic：lr ≤ 0 |
| [`AdEMAMix`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#AdEMAMix) / [`NewAdEMAMix`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewAdEMAMix) | `struct{ LR, Beta1, Beta2, Beta3, Alpha, Eps float32; Warmup int }` / `NewAdEMAMix(lr, b1, b2, b3, alpha, warmup, eps)` | Adam 外加第二条刻意**不校正**的慢速梯度 EMA，以 α 混入（arXiv:2409.03137，ICLR 2025）；α 线性/β3 半衰期 warmup 调度；不含解耦 weight decay；状态按参数指针键控 | panic：lr ≤ 0、beta ∉ [0,1)、alpha < 0、warmup < 0、eps ≤ 0；Step：参数尺寸变更 |
| [`NewAdEMAMixDefault`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewAdEMAMixDefault) | `NewAdEMAMixDefault(lr, warmup) *AdEMAMix` | 论文默认值：0.9 / 0.999 / 0.9999 / α=5 / 1e-8 | panic：lr ≤ 0 |
| [`ScheduleFreeAdamW`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#ScheduleFreeAdamW) / [`NewScheduleFreeAdamW`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewScheduleFreeAdamW) | `struct{ LR, Beta1, Beta2, Eps, WeightDecay float32; WarmupSteps int }` / `NewScheduleFreeAdamW(lr, b1, b2, eps)` | 免调度 AdamW（arXiv:2405.15682，NeurIPS 2024）：在 `y` 处求梯度、在 `z` 上跑基础 AdamW、可部署权重是平均后的 `x`；**训练中参数恒持 `y`**——`Eval`/`Train` 转换；`WeightDecay` 解耦、在 `y` 处施加；偏差校正随官方 v1.3+ | panic：lr ≤ 0、b1 ∉ (0,1)、b2 ∉ [0,1)、eps ≤ 0；Step：eval 模式参数（按编号指名）或尺寸变更；Train/Eval：nil 数据或尺寸变更 |
| [`NewScheduleFreeAdamWDefault`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#NewScheduleFreeAdamWDefault) | `NewScheduleFreeAdamWDefault(lr) *ScheduleFreeAdamW` | 默认值 β1 0.9、β2 0.999、eps 1e-8（官方 LR 指引：调度版 AdamW 的 1×–10×） | panic：lr ≤ 0 |
| [`Train`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#ScheduleFreeAdamW.Train) / [`Eval`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#ScheduleFreeAdamW.Eval) | `(*ScheduleFreeAdamW).Train(params)` / `Eval(params)` | 对每个持有状态的参数做原位 y↔x 转换（幂等；无状态参数不动）：`Eval` 用于评估/导出，`Train` 用于下一次 `Step` 之前 | panic：nil 数据、参数尺寸变更 |

### 状态持久化（`"LNO1"` 流——只报 error，绝不 panic）

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`SaveState`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#SaveState) | `SaveState(w, o Optimizer, params) error` | 逐参数状态，**按下标键控**——Load 必须同序；超参数不入流；SGD → 23 字节恒等流；字节确定性；kind 0–4 = SGD/Momentum/Adam/AdEMAMix/ScheduleFreeAdamW（kind 4 另携带每个参数的 train/eval 模式位） | error：不支持的优化器类型、nil 参数、I/O |
| [`LoadState`](https://pkg.go.dev/github.com/qorm/LNN/optimizer#LoadState) | `LoadState(r, o Optimizer, params) error` | 先全量校验再应用：失败时 `o` 原样不动；在场记录覆盖、缺席记录删除、陈旧键保留；续训位级一致（eval 模式的 ScheduleFree 流会恢复 eval 模式——`Step` 随之以 panic 守门直至 `Train`） | error：magic/version/kind 错误、计数不符、形状不符、pow 不一致（Adam/AdEMAMix/ScheduleFree）、`t`/`k` > 2²⁴、`lrMax`/`wsum` 非有限、mode 字节非法、截断（`io.ErrUnexpectedEOF`） |

## serialize — [godoc](https://pkg.go.dev/github.com/qorm/LNN/serialize) · [持久化](persistence.md)

`"LNNS"` 线格式：magic、version、count，每个张量的 rank + shape +
小端 float32 载荷，以及——v2——覆盖全部内容的尾部 CRC-32C 校验和
（无校验和的 v1 仍可读；v2 是唯一写布局）。例外域：加载把输入视为
不可信字节流——一切失败都是 error，绝不 panic，恶意流的分配量只与
实际送达的字节成比例。

| 符号 | 签名 | 语义 | 失败 |
|---|---|---|---|
| [`Version`](https://pkg.go.dev/github.com/qorm/LNN/serialize#Version) | `const Version uint8 = 2` | 本构建写入的格式版本（v2，带 CRC-32C 校验和）；v1 仍可读以兼容遗留检查点；未知版本一律拒绝，绝不猜测 | — |
| [`WriteTensors`](https://pkg.go.dev/github.com/qorm/LNN/serialize#WriteTensors) | `WriteTensors(w, ts []*tensor.Tensor) error` | 编码张量切片（v2：末尾追加 CRC-32C 校验和） | error：nil 张量、rank > 8、数量 > 2²⁰、负维度、Shape/Data 不符、溢出、I/O |
| [`ReadTensors`](https://pkg.go.dev/github.com/qorm/LNN/serialize#ReadTensors) | `ReadTensors(r) ([]*tensor.Tensor, error)` | 解码（v1 与 v2 均可读；v2 校验校验和）；分配前先过固定上限；知长读取器预先核对剩余字节，未知长读取器渐进增长 | error：magic 错误、版本偏移（带方向提示）、超限、截断（`io.ErrUnexpectedEOF`）、尾随字节、v2 校验和失配 |
| [`WriteParameters`](https://pkg.go.dev/github.com/qorm/LNN/serialize#WriteParameters) | `WriteParameters(w, params []*autograd.Variable) error` | 对 `p.Data` 依次 `WriteTensors`（顺序 = 加载键） | error：nil 参数/Data、WriteTensors 的各类错误 |
| [`LoadParameters`](https://pkg.go.dev/github.com/qorm/LNN/serialize#LoadParameters) | `LoadParameters(r, params []*autograd.Variable) error` | **原位**复制回参数（指针身份不变）；先全量校验形状；**陈旧 `Grad` 刻意保留**——复用前先 `ZeroGrad` | error：nil 参数、数量/形状不符、ReadTensors 的各类错误 |

## 概念交叉索引

| 如果你的问题是…… | 读 |
|---|---|
| “这个算子输出什么形状？” | [shapes-and-broadcasting.md](shapes-and-broadcasting.md)——完整广播表、不对称归约 |
| “怎么训练？” | [training.md](training.md)——四段式循环、梯度裁剪、热替换、指针键控状态 |
| “怎么存档？这个文件格式安全吗？” | [persistence.md](persistence.md)——逐字节的 `"LNNS"`/`"LNO1"`、位级一致续训、不可信流契约 |
| “LTC 到底在算什么？” | [ltc.md](ltc.md)——方程对代码、稀疏收缩、参数表 |
| “CfC 与 LTC 有何不同？” | [cfc.md](cfc.md)——Lemma 1 闭式解、exprel 稳定化 |
| “各层如何组合？” | [architecture.md](architecture.md)——tensor → autograd → nn、操作标记式反向 |
| “生产环境会踩什么坑？” | [pitfalls.md](pitfalls.md)——并发、float32 溢出、重复 Backward、极小 `ts` |

指南总索引：[doc/README.md](README.md)。仓库根 `README.md` 有快速上手；
`examples/` 是完整可运行的训练循环。
