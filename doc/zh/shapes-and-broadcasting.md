> [English](../shapes-and-broadcasting.md) | 中文

# 形状与广播

**摘要：** lnn 使用稠密行主序形状，配一个显式枚举的广播（broadcasting）子集（五条规则，其他一律 panic）；其归约输出形状*并非*完全对称——依赖之前请先读下面的表格。

**读者对象：** 任何编写前向或损失代码的人；这是所有"输出是什么形状？"问题的参考手册。

## 布局与秩

`Shape []int` + 扁平行主序 `Data []float32`；`[m, n]` 张量的元素 `(i, j)` 位于 `Data[i*n+j]`。本库聚焦 1D/2D：

| 仅限 2D（其他秩 panic） | 任意秩 |
|---|---|
| `MatMul`、`Transpose`、`Rows`、`Cols`、`ConcatCol`、`SliceCol`、`SliceRow`、`SoftmaxRows`、`LogSoftmaxRows`、`SumRows`、`SumCols` | `Add`、`Sub`、`Hadamard`、`Scale`、`Neg`、`Apply`、`Tanh`、`Sigmoid`、`Exp`、`Log`、`Pow`、`Softplus`、`Clip`、`SumAll`、`MeanAll` |

`MatMul` 只是矩阵乘法：`[m, k] × [k, n] → [m, n]`，秩或内维不匹配即 panic。

## 广播规则（完整列表）

二元逐元素算子（`Add`、`Sub`、`Hadamard`）恰好接受以下组合——实现为 `tensor/ops.go`（`broadcastShape`）里的一个 `switch`，而非通用的 NumPy 式广播：

| # | a | b | 结果 | 示例 |
|---|---|---|---|---|
| 1 | 形状 S | 相同形状 S | S | `[2,3] ⊕ [2,3] → [2,3]` |
| 2 | 标量（恰含一个元素的任意张量） | 任意 | b 的形状 | `[1] ⊕ [2,3] → [2,3]`；`[1,1]` 也是标量 |
| 3 | 任意 | 标量 | a 的形状 | `[2,3] ⊕ [1] → [2,3]` |
| 4 | `[m, n]` | 行向量 `[n]` 或 `[1, n]`（n 必须匹配） | `[m, n]` | `[2,3] ⊕ [3] → [2,3]` |
| 5 | 行向量 | `[m, n]` | `[m, n]` | `[3] ⊕ [2,3] → [2,3]` |
| 6 | `[m, 1]` 列向量 | `[m, n]`（行数相同） | `[m, n]` | `[2,1] ⊕ [2,3] → [2,3]` |
| 7 | `[m, n]` | `[m, 1]` 列向量 | `[m, n]` | `[2,3] ⊕ [2,1] → [2,3]` |
| 8 | `[m, 1]` | 行向量 `[n]` 或 `[1, n]` | **外积** `[m, n]` | `[2,1] ⊗ [3] → [2,3]` |
| 9 | 行向量 | `[m, 1]` | **外积** `[m, n]` | `[3] ⊗ [2,1] → [2,3]` |

其他所有组合都会 panic：

```
tensor: shapes [2 3] and [2 2] are not broadcastable
```

特别地，长度*不同*的两个 1D 张量永远无法广播（`[2] ⊕ [3]` panic），元素数相同但布局不同的也不行（`[1,6] ⊕ [2,3]` panic）。

### 输出形状的提升

实现带来的两个需要知道的后果：

- **1D 对 1D 的结果被提升为 `[1, n]`。** `Add([n], [n])` 返回形状 `[1, n]`，而不是 `[n]`（规则 1 解析为 `[n]`，随后二元算子驱动器把 1D 输出提升为 2D）。
- **标量 × 标量：结果仍是单元素，但形状取决于操作数的秩。** 两个 0 维操作数（形状 `[]`，即无参数的 `tensor.New()`）产出形状 `[1]`；但两个 `[1]` 张量走规则 1 再经 1D 提升，`[1] ⊕ [1] → [1, 1]`。换言之"单元素 × 单元素得到单元素结果"恒成立，形状却不一定是 `[1]`。（甚至存在顺序差异：`Add(tensor.New(1), tensor.New())` 走标量快速路径得 `[1]`，而 `Add(tensor.New(), tensor.New(1))` 得 `[1, 1]`——别依赖 0 维张量的边角行为。）

一眼即可验证（每行都是对 `m := [2,3]` 的真实调用）：

```go
tensor.Add(m, m).Shape                        // [2 3]   （规则 1）
tensor.Add(m, scalar).Shape                   // [2 3]   （规则 2/3）
tensor.Add(m, rowVec).Shape                   // [2 3]   （规则 4/5）
tensor.Add(m, colVec).Shape                   // [2 3]   （规则 6/7）
tensor.Hadamard(colVec, rowVec).Shape         // [2 3]   （规则 8/9，外积）
tensor.Add(tensor.New(3), tensor.New(3)).Shape // [1 3]  （1D⊕1D 提升）
tensor.Add(tensor.New(), tensor.New()).Shape   // [1]    （0 维⊕0 维）
tensor.Add(tensor.New(1), tensor.New(1)).Shape // [1 1]  （[1]⊕[1] 提升）
```

## 归约形状（速查）

对输入 `[m, n]`：

| 算子 | 轴 | 输出形状 | 备注 |
|---|---|---|---|
| `SumAll` | 全部 | `[1]` | 空张量得 `0` |
| `MeanAll` | 全部 | `[1]` | 空张量 **panic**（零个元素的均值无定义） |
| `SumRows` | 0（跨行） | **`[1, n]`** | 保持 2D，以便重新广播回 `[m, n]` |
| `SumCols` | 1（跨列） | **`[m]`** | 1D |
| `SoftmaxRows` / `LogSoftmaxRows` | 逐行 | `[m, n]` | 减去最大值，数值稳定；零列输入产出同形状的空结果 |
| `SliceRow(i)` | — | `[1, n]` | 拷贝 |
| `SliceCol(from, to)` | — | `[m, to-from]` | 拷贝；非法/空区间 panic |
| `autograd.Col(i)` | — | `[m, 1]` | `SliceCol(i, i+1)` |
| `autograd.GatherRows(a, idx)` | — | `[m]`（1D） | `len(idx)` 必须等于行数 |
| `ConcatCol(...)` | 沿列 | `[m, Σn]` | 所有输入共享 `m` |
| `Transpose` | — | `[n, m]` | |
| `MatMul(a, b)` | — | `[m, n]` | 来自 `[m, k] × [k, n]` |
| `Stack(k 个形状 S 的张量)` | 新首维 | `[k, S...]` | **实验性：** 对 2D 输入产出 3D，而其他任何算子都不消费 3D（秩 ≠ 2 即 panic） |

### 坦诚的部分：SumRows 和 SumCols 不对称

`SumRows` 返回 `[1, n]` 而 `SumCols` 返回 `[m]`——一个保持矩阵，另一个降为向量。这没有数学上的理由；纯属历史：约定随代码演化，而 `autograd` 的反向传播和 `nn` 的内部实现都建立其上（例如 `LogSoftmaxRows` 的梯度会把 `SumCols` 结果重塑为 `[m, 1]`，而偏置梯度依赖 `SumRows` 保持行可广播）。现在改动任一形状都将是跨整个模块的 API 破坏性变更，因此它作为一个单独的评估项被追踪，而不是就地修复（路线图——见 [pitfalls.md](pitfalls.md)）。在此之前：请核对上表，并在新代码中优先使用 `autograd` 算子（它们在内部处理归约），而非手动的 `tensor` 归约。

同样的沿革也解释了 1D⊕1D → `[1, n]` 的提升：1D 参数的叶梯度仍然以叶自身的形状返回（下面的 `SumToShape` 归约到*操作数*的形状），所以这个提升很少泄漏到用户代码里——但它确实会出现在中间的 `Data.Shape` 值中。

## 反向归约：SumToShape

广播算子用 `tensor.SumToShape(grad, shape)` 把输出梯度归约回操作数形状。对形状 `[m, n]` 的梯度：

| 目标形状 | 归约 | 结果 |
|---|---|---|
| `[m, n]` | 恒等（克隆） | `[m, n]` |
| `[1]`（标量） | `SumAll` | `[1]` |
| `[n]` | 列和（`SumRows`，重塑为 1D） | `[n]` |
| `[1, n]` | 列和 | `[1, n]` |
| `[m, 1]` | 行和（`SumCols`，重塑） | `[m, 1]` |
| 其他 | — | **panic** |

正是这一点，使得叶梯度即使在前向过程中发生过广播（包括那些前向输出被提升为 `[1, n]` 的 1D 叶），也永远与叶的形状一致。
