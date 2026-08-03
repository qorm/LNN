# LNN 常见问题（FAQ）

> [English](../faq.md) | 中文

常见问题，简短回答——每问都指向承载完整解释的指南，有用处还附
实测代码片段（更长的任务式食谱见 [cookbook.md](cookbook.md)）。

- [loss 不降怎么办？](#loss-不降怎么办)
- [出现 NaN 损失怎么办？](#出现-nan-损失怎么办)
- [ts 怎么选？](#ts-怎么选)
- [units 与 unfolds 怎么选？](#units-与-unfolds-怎么选)
- [大 units 模型内存不够怎么办？](#大-units-模型内存不够怎么办)
- [为什么 Backward 后梯度还在累加？](#为什么-backward-后梯度还在累加)
- [为什么 Step 不 ZeroGrad？](#为什么-step-不-zerograd)
- [LTC 和 CfC 该用哪个？](#ltc-和-cfc-该用哪个)
- [不同机器上结果末位不同正常吗？](#不同机器上结果末位不同正常吗)
- [加载报错 "stream holds model kind …" 怎么读？](#加载报错-stream-holds-model-kind--怎么读)
- [如何用 Adam 续训？](#如何用-adam-续训)

---

## loss 不降怎么办？

**简答：** 按 [training.md](training.md) 末尾的症状表逐条排查
——它按各成因的出现频率排序。常见嫌疑：lr 过大（loss → `NaN`）
或过小（缓慢爬行）、漏了 `ZeroGrad` 或重复 `Backward`（梯度恰好
翻倍）、LTC 上没有梯度裁剪（`1/(den+eps)` 除法产生梯度尖峰）、
在前向与 `Backward` 之间改了参数 `Data`。

然后用 [cookbook.md](cookbook.md) 食谱 6 做一次仪表化：打印按参数
梯度范数并扫描 `NaN`/`Inf`。范数恰好是预期的 `2×`，是重复
`Backward`/漏 `ZeroGrad` 的指纹；已接线模块上 nil 的 `Grad` 是
建图 bug；单个参数上的巨大范数指出裁剪该在哪里生效。

## 出现 NaN 损失怎么办？

**简答：** 找到产生它的算子，然后——如果用了 Adam 或 Momentum
——假定优化器状态已被毒化并重建它。完整的溢出表在
[pitfalls.md](pitfalls.md) §2。

让 `NaN` 在此严重的两个事实：

1. 没有全局溢出防护。`Exp(x)` 在 `x ≳ 88` 溢出，`Log(x)` 对
   `x ≤ 0` 不加检查地给出 `NaN`/`−Inf`，`Div(a, 0)` 是 `±Inf`，
   而一旦有一个元素是 `NaN`，它就会乘着逐元素路径走完本轮迭代
   的剩余部分。
2. **进入优化器状态的 `NaN` 是永久性的。** Momentum 的速度与
   Adam 的矩是滚动累加器；一个 `NaN` 梯度就会毒化它们，后续的
   健康梯度与 `ZeroGrad` 都冲洗不掉——毒物存在于优化器的缓冲
   里，而不在 `Grad` 里。实测：在 Adam 运行的第 3 轮注入一个
   `NaN` 梯度：

```go
w := autograd.Var(tensor.FromData([]float32{1}, 1))
opt := optimizer.NewAdamDefault(0.1)
for it := 0; it < 25; it++ {
	loss := autograd.Pow(autograd.Sub(w, autograd.Const(tensor.FromData([]float32{3}, 1))), 2)
	w.ZeroGrad()
	loss.Backward()
	if it == 3 {
		w.Grad.Data[0] = float32(math.NaN()) // 一个被毒化的梯度
	}
	opt.Step([]*autograd.Variable{w})
	// ...
}
```

```
iter  0  w=1.1
iter  4  w=NaN
iter  8  w=NaN
iter 12  w=NaN
iter 16  w=NaN
iter 20  w=NaN
iter 24  w=NaN
```

唯一的补救是丢弃优化器（全新状态）或重置受影响的参数。预防
正是"裁剪（[training.md](training.md) 梯度裁剪一节）+ 保持
logit 有界"的论据：`NaN` 一旦进入有状态优化器，那份状态就只能
推倒重来。

## ts 怎么选？

**简答：** 把一个时间单位锚定到某个物理量（通常选采样间隔），
传入每步覆盖的时间跨度——若每个序列步对应一个采样间隔，就取
`ts = 1.0`。`ts ≳ 1e-3` 保持完整物理保真度；`ts` 必须为正且
有限，否则 `Step` panic。完整契约与有限性域见 [ltc.md](ltc.md)
的 ts 一节。

`ts` 的作用——在零初态、输入 `x = 1` 下单步 CfC 的实测（LTC
行为相同——以其 `unfolds` 做欧拉）：

```
ts= 0.01  out[0]=+0.007009  |state|=+0.007009
ts= 0.10  out[0]=+0.064129  |state|=+0.064129
ts= 1.00  out[0]=+0.304690  |state|=+0.304690
ts=10.00  out[0]=+0.351644  |state|=+0.351644
```

小 `ts` 几乎不推进膜电位；大 `ts` 让它弛豫向稳态（输出饱和）。
若事件以不规则间隔到达，逐步驱动 `ts`——见 [cookbook.md](cookbook.md)
食谱 5。

## units 与 unfolds 怎么选？

**简答：** `units` 是模型容量（从小处起步——液态细胞所需单元
远少于经典 RNN）；`unfolds` 只是 LTC 的积分器精度（ncps 默认
为 6；增大它以图成本线性增长为代价收紧欧拉求解）。CfC 没有
`unfolds`——它的闭式步在时间跨度上是常量成本。

实测的成本：

| 成本 | 公式 / 数值 |
|---|---|
| 细胞参数 + 掩码（全接线） | `O(units²)`——加载/构建峰值 `92·U²` 字节，`U = units = inDim`（[persistence.md](persistence.md)）；`U = 2048` ≈ 368 MiB |
| 构建全接线 `units = 1024` 细胞 | 稀疏收缩（sparse contraction）之后实测 ~32 MB（LTC 36.4 MiB / CfC 32.4 MiB）——不再是旧设计的 ~8 GiB 指示矩阵悬崖 |
| LTC 训练图（每序列） | `∝ units × unfolds × seqLen` 个激活块保留到 `Backward`（[pitfalls.md](pitfalls.md) §9） |
| CfC 训练图（每序列） | `∝ units × seqLen`——无 `unfolds` 因子 |
| 每步墙钟（wall-clock） | LTC 随 `unfolds` 增长；CfC 与时间跨度无关 |

因此：`units` 保持适度，`unfolds` 无理由就取 4–8；若
`seqLen × unfolds` 把图撑爆，换 CfC 或缩短展开。

## 大 units 模型内存不够怎么办？

**简答**，按收益排序：

1. **换 CfC**——它把 `unfolds` 因子从训练图中整个拿掉
   （[cfc.md](cfc.md)）。同一 ODE、同样 13 个参数、解析积分器。
2. **稀疏化接线。** `nn.RandomSparse(inDim, units, sensoryP,
   recurrentP, rng)` 以 `1−p` 的概率独立丢弃突触；未接线的突触
   在算术上是中性的，因此稀疏收缩只为存在的接线付费
   （[ltc.md](ltc.md) 接线一节）。掩码在构建时抽取一次，不可变。
3. **缩短展开窗口**——图把每个展开步的一切中间张量保留到
   `Backward`（[pitfalls.md](pitfalls.md) §9）；在更短片段上做
   截断 BPTT 是标准的缓解手段。
4. 然后才考虑减小 `units`。

```go
// 稀疏细胞：约 50% 的突触存在，构建时抽取一次
wiring := nn.RandomSparse(4, 256, 0.5, 0.5, rng)
cell := nn.NewLTC(4, 256, wiring, 6, rng)
cfc := nn.NewCfC(4, 256, wiring, rng) // 同一接线两种细胞都适用
```

（`NewLTC`/`NewCfC` 本身不受内存限额——`2048` 的 `units`/`inDim`
上限只在加载路径上，因为加载的输入是不可信流；见
[persistence.md](persistence.md)。）

## 为什么 Backward 后梯度还在累加？

**简答：** 因为叶梯度跨 `Backward` 调用**累加**——跨不同的图
也是——直到你 `ZeroGrad`。这是核心梯度语义，不是泄漏：食谱 3
（梯度累积）正是建立在这之上。

```go
a := autograd.New([]float32{1, 2, 3}, 3)
y := autograd.SumAll(autograd.Hadamard(a, a)) // dy/da = 2a
y.Backward()
fmt.Println(a.Grad.Data) // [2 4 6]
y.Backward()             // 同一张图，第二次
fmt.Println(a.Grad.Data) // [4 8 12]——恰好翻倍
```

标准循环先清零：对每个参数 `ZeroGrad` → 前向 → 一次 `Backward`
→ 更新。对*同一张*图调用两次 `Backward` 恰好让梯度翻倍（三次
调用三倍——[pitfalls.md](pitfalls.md) §3）。如果梯度看起来是
应有值的整数倍，数一数你的 `Backward` 与 `ZeroGrad` 调用。

## 为什么 Step 不 ZeroGrad？

**简答：** 因为清零是*你的*契约，一个替你清零的优化器会悄无声息
地破坏梯度累积。

累加语义是本库的梯度契约；`optimizer.Step` 刻意只做更新阶段
（[training.md](training.md) 的 Step 契约一节）。普通训练每轮
迭代前清零；每 `N` 轮清零一次、同时在每个微批后都 `Backward`，
同一个 `Step` 调用就免费给你梯度累积——[cookbook.md](cookbook.md)
食谱 3 实测其以 `3.6e-7`（float32 加法顺序）复现全批量梯度。

## LTC 和 CfC 该用哪个？

**简答：** 同一 ODE，两种积分器——按成本选，不按表达能力选。
CfC 的闭式步图成本为常量（无 `unfolds`）；LTC 的欧拉循环每步
花费 `unfolds` 个子步，但它是参考实现的精确格式。内存或长序列
吃紧时从 CfC 起步；要复现 ncps 动力学时选 LTC。其余一切共享
——参数、接线、`ts` 契约——并且可以在 `nn.Cell` 背后直接互换
（同一 seed 甚至给出逐位一致的初始化）。

决策表、互换示范与实测对比：[cookbook.md](cookbook.md) 食谱 8。
两种细胞各自的论文↔代码对照：[ltc.md](ltc.md)、[cfc.md](cfc.md)。

## 不同机器上结果末位不同正常吗？

**简答：** 同架构：任何差异都不应接受——同一 seed 在那里逐位
一致，本库自己的测试钉死位模式。跨架构（arm64 对 amd64）：每次
融合乘加（FMA）缩合可差至 1 ULP，且链上会累积（CfC 的
Box-Muller 初始化实测 ≤ 6 ULP）。这是 Go 的行为——语言允许把
多个浮点运算融合为一次舍入——不是本库的缺陷。

黄金向量测试所用的分级（[persistence.md](persistence.md) 黄金
向量一节）：

| 对象 | 保证 |
|---|---|
| 线上格式布局（magic、version、形状、计数、字节序） | **任何**平台逐字节冻结 |
| 浮点载荷，同平台/工具链 | 逐位（`Float32bits`，含 `NaN`/`−0`） |
| 浮点载荷，其他架构 | 16 ULP 窗口内（约为实测最大值的 2.7 倍；仍紧到 32 ULP 会作为损坏失败） |

因此：在同一台机器上断言时用 `math.Float32bits` 比较位模式，
跨机器允许一个小的 ULP 窗口，绝不字符串比较打印出来的十进制
——让同机运行可复现的种子纪律见 [cookbook.md](cookbook.md)
食谱 12。

## 加载报错 "stream holds model kind …" 怎么读？

**简答：** 加载路径上的一切失败都是 `error`（绝不 panic），而
消息告诉你它属于哪个桶。读前缀：`nn:` = 模型级，`serialize:` =
张量流级，`optimizer:` = 状态流级。

| 消息 | 含义 | 处置 |
|---|---|---|
| `nn: stream holds model kind 1 (CfC), not LTC (kind 0)` | 用错加载器 | 改用 `nn.LoadCfC`——消息会点名文件实际是什么 |
| `serialize: bad magic … not an LNN tensor stream` | 根本不是 LNN 文件 | 检查是不是指错了文件 |
| `serialize: unsupported format version 99 (this build reads version 2): … newer version …` | 由更新的库写出 | 升级库 |
| `…: no earlier layout exists, the stream is corrupt or forged` | version 字节低于 1 | 损坏——拒绝 |
| `…: truncated stream: claims N data bytes but only M remain: unexpected EOF` | 文件被截短（用 `errors.Is(err, io.ErrUnexpectedEOF)` 匹配） | 重新传输；失败的加载不会动你的模型 |
| `nn: LTC header has units=4096, exceeding the load limit 2048` | 头超过加载路径上限（另有 `unfolds` 1024、`inDim` 2048） | 恶意或过大——检查在任何 blob 分配之前就已触发 |

以上所有消息都出自 [cookbook.md](cookbook.md) 食谱 10 的真实
输出，那里有完整的"分类-处置"范式。其背后的契约——固定限额、
先校验后分配、失败零副作用——在 [persistence.md](persistence.md)
的不可信流安全契约一节。

## 如何用 Adam 续训？

**简答：** 三条流，不是一条——模型、其余模块的参数、优化器
状态；然后把三者都加载进全新对象，并保持超参一致。
[cookbook.md](cookbook.md) 食谱 4 是完整程序；在那里实测：50 步
→ 检查点 → 续训 50 步，与不间断的 100 步运行**逐位**吻合
（逐步 loss 与最终参数，`Float32bits`）。

```go
// 保存
must(nn.SaveCfC(modelFile, cell))                        // 细胞（LTC 用 SaveLTC）
must(serialize.WriteParameters(paramFile, readout.Parameters()))
must(optimizer.SaveState(stateFile, opt, params))        // Adam 的 m、v、t、偏差校正幂

// 续训
loaded, err := nn.LoadCfC(modelFile)
readC := nn.NewLinear(units, 1, rng)                     // seed 无关紧要
must(serialize.LoadParameters(paramFile, readC.Parameters()))
optC := optimizer.NewAdamDefault(lr)                     // 同一组超参
must(optimizer.LoadState(stateFile, optC, nn.ParametersOf(loaded, readC)))
```

会咬人的细节：仅有模型检查点会悄无声息地重置 Adam 的自适应
（偏差校正从 `t = 0` 重新开始）；Load 时的 `params` 顺序必须与
Save 时一致（状态以下标为键）；超参刻意不在 `"LNO1"` 流里
——`LoadState` 会用保存的 `Beta^t` 幂校验目标优化器，因此 beta
不匹配会高声失败。格式与契约：[persistence.md](persistence.md)
优化器状态一节。
