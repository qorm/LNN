> [English](../pitfalls.md) | 中文

# 陷阱、边界与残余风险

**摘要：** 本库*不*替你防护的一切，从红队审计提炼为用户须知：单线程契约、`float32` 溢出、重复 `Backward` 的精确语义，以及技术债路线图。

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

稳定化的内部实现只存在于上表所列之处（外加 `Sigmoid`，以及 LTC 的 `ts` 缩放——见 [ltc.md](ltc.md)）。其余都是你自己的问题：保持 logit 有界、钳制 `Log` 的输入、裁剪梯度（[training.md](training.md)）。一旦存在一个 `NaN`，它就会在本轮迭代的剩余时间里沿逐元素路径扩散。

**一个需要知道的非对称 NaN 行为：** `MatMul` 在内层循环里跳过零乘数（`tensor/ops.go:20`），因此 `0 * NaN` 贡献 `0` 而不是毒化乘积：

```go
tensor.MatMul(tensor.FromData([]float32{0}, 1, 1),
	tensor.FromData([]float32{float32(math.NaN())}, 1, 1)).Data // [0]，而不是 [NaN]
```

不要指望 MatMul 能把稀疏位置上的 NaN 暴露出来。

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

少数反向闭包在反向时读取父节点的 `Data`（例如 `Log` 从父节点的*当前*值计算 `1/x`），而 `MatMul` 的反向使用保存的操作数张量。在前向与 `Backward` 之间修改叶节点会悄无声息地改变梯度：

```go
x := autograd.New([]float32{2}, 1)
l := autograd.SumAll(autograd.Log(x)) // d/dx = 1/x，在 Backward 时求值
x.Data.Data[0] = 8                    // Backward 之前的修改
l.Backward()
fmt.Println(x.Grad.Data) // [0.125] = 1/8，而不是 1/2
```

因此原地参数更新必须严格放在 `Backward` *之后*。

## 5. GatherRows 会拷贝索引（已修复的隐患）

`autograd.GatherRows(a, idx)` 在入口处拷贝 `idx`（`autograd/ops.go:271`），因此调用方可以在前向与反向之间随意复用或修改该切片——梯度由前向所用的索引算出：

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

每个中间张量都保持存活，直到 `Backward` 完成。一次 LTC step 展开 `unfolds` 轮 ODE 迭代、每轮 O(units²) 个突触算子；`T` 步的序列会把这一切再乘以 `T`。内存随每次迭代的算子数增长，而不仅仅随参数增长——`units`、`unfolds` 和序列长度都要保持适度。

## 路线图与技术债

公开追踪；在文档所述范围（单线程、适度规模、合理的 `ts`/`lr`、调用方裁剪梯度）内，没有一项阻碍生产使用：

| 条目 | 状态 |
|---|---|
| `autograd.Div` 闭式化 | **已完成：** 单图节点 + 商法则反向（`da = g/b`、`db = −g·a/b²`，`autograd/ops.go:171-194`）。注意小除数固有的 `1/b²` 梯度放大依然存在——以 LTC 的 `eps = 1e-8` 下限计，最高约 `1e16` 倍——因此仍建议梯度裁剪 |
| 统一归约形状约定（`SumRows`/`SumCols`、1D 提升） | 单独评估；API 破坏性 |
| `tensor.Stack` | 实验性：产出没有其他算子消费的 3D 张量（`tensor/tensor.go:165`）；为兼容保留 |
| CfC（Closed-form Continuous-time）细胞 | 未实现 |
| 内置优化器 | 未实现；手写 SGD 是受支持的范式 |
| 序列化（Save/Load） | 未实现；参数就是朴素的 `[]float32` 缓冲区——请自行快照 `p.Data.Data` |
| `make test` 之外的基准/CI 工具 | 进行中 |
