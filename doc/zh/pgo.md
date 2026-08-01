# PGO（profile-guided optimization，画像引导优化）

> [English](../pgo.md) | 中文

**摘要：** Go 的画像引导优化（PGO）能为基于本库构建的二进制带来什么、不能带来什么——附在本仓库基准套件上的实测数字、这些数字共同悬于一线的单个编译器决策，以及给*你自己的*二进制采集画像的完整工作流（画像只能放在你那边：LNN 是库，不附带任何画像）。

**读者对象：** 追求吞吐量的用户；纠结要不要在 `main` 包旁放一个 `default.pgo` 的人。

本文涉及的工具链事实均出自 Go 官方 PGO 文档 <https://go.dev/doc/pgo>。测量命令已封装为 `make pgo-profile` / `make pgo-bench` 目标（见下文）。

## 一句话版本

- PGO 自 Go 1.20 起可用；自 Go 1.21 起，`go build` 会自动拾取 main 包目录下的 `default.pgo`（`-pgo=auto`）；`-pgo=<路径>` 可显式指定画像（[go.dev/doc/pgo](https://go.dev/doc/pgo)）。
- 截至 Go 1.22，Go 官方在一组代表性 Go 程序上报告的提升为**约 2–14%**（同一出处）。对真实应用程序，请以此为预期——而不是下面最大的那个数字。
- 本机实测（Go 1.26.5，Apple M4）：当画像恰好翻转了某一个内联决策时，逐元素张量算子 ns/op 下降 **−57 ~ −70%**，`nn` 细胞基准下降 **约 −4 ~ −6%**（占比小于融合前的 −7%：融合 LTC 内核的热循环不再经过被内联的包装函数）；没翻转时，收益 **≈ 0**。两种结果都在下文中如实给出——效应为真，但*双态*，且不受用户直接控制。
- PGO 绝不改变数值：分配次数逐项不变，`examples/cfc-sequence` 的输出在有/无画像两种构建下逐位一致（已验证，见下）。

## 为什么库本身不附带画像

PGO 是**按 main 包**应用的：画像要么是放在*你的* `package main` 旁边的 `default.pgo`，要么是传给 `go build -pgo=` 的路径。库没有 main 包，附带画像无处可挂。而且这个优化是全程序范围的——按 Go 官方文档的说法，"PGO in Go applies to the entire program … including packages in dependencies. This means that the unique way your application uses a dependency impacts the optimizations applied to that dependency."（PGO 作用于整个程序……包括依赖里的包；你的应用使用某个依赖的独特方式会影响该依赖被如何优化。）真正有用的画像，是从*你的*工作负载里采的——那只有你才有。

## 在本仓库上的实测

环境：`go1.26.5 darwin/arm64`，Apple M4（10 核，16 GB），macOS 26.5.2（Darwin 25.5.0）。方法：从两个 `nn` 基准采集画像（`go test ./nn -run '^$' -bench 'BenchmarkLTCStep|BenchmarkUnrollBackward' -benchtime=2s -cpuprofile=…`），然后完整 17 个基准在 `-pgo` 开/关下交错 A/B/A/B 运行，每轮 `-count=3`（每格 n = 6），带 `-benchmem`。全部 17 个基准的平均 ns/op（在融合后的树上实测；`tensor`/`autograd` 基线列与本表融合前版本在轮间噪声内一致，两个头条 `nn` 行为融合内核基线）：

| 基准 | 基线 | PGO | Δ ns/op | 分配 |
|---|---:|---:|---:|---|
| tensor/AddBroadcastRow | 30,970 | 9,388 | **−69.7 %** | 不变 |
| tensor/Hadamard | 20,288 | 8,679 | **−57.2 %** | 不变 |
| autograd/ChainForwardBackward | 401,864 | 319,478 | **−20.5 %** | 不变 |
| autograd/DivDenLoop | 297,555 | 273,970 | −7.9 % | 不变 |
| autograd/GatherRowsBackward | 10,590 | 9,765 | −7.8 %（t = 1.1，在基线散布之内） | 不变 |
| nn/UnrollPeakMemory512 | 69,342,589 | 64,999,033 | −6.3 % | 不变 |
| nn/UnrollRematCfC | 1,869,180 | 1,755,819 | −6.1 %（t = 1.9，在散布之内） | 不变 |
| nn/UnrollRemat | 3,447,318 | 3,241,136 | −6.0 % | 不变 |
| nn/UnrollRematPeakMemory512 | 165,574,634 | 156,075,163 | −5.7 % | 不变 |
| nn/LTCStep | 90,618 | 87,395 | −3.6 % | 不变 |
| nn/UnrollBackward | 1,376,367 | 1,327,337 | −3.6 % | 不变 |
| tensor/MatMul64 | 76,079 | 74,341 | −2.3 % | 不变 |
| tensor/MatMul128 | 647,442 | 636,798 | −1.6 % | 不变 |
| tensor/SumCols | 7,378 | 7,313 | −0.9 % | 不变 |
| tensor/SoftmaxRows | 44,036 | 45,828 | +4.1 %（在散布之内） | 不变 |
| tensor/SumRows | 8,262 | 7,405 | −10.4 %（基线有一个离群点；中位数 7,642 → 7,411） | 不变 |
| tensor/Transpose | 10,482 | 11,093 | +5.8 %（在轮间散布之内） | 不变 |

五个大幅移动的基准的 Welch t 统计量（n = 6 对 6）：AddBroadcastRow 64.3、Hadamard 73.7、ChainForwardBackward 10.7、UnrollRemat 5.6、DivDenLoop 5.1——全部显著。LTCStep（2.1）与 UnrollBackward（2.9）处于边缘；GatherRowsBackward（1.1）、UnrollRematCfC（1.9）、MatMul 对（约 1.6）以及 SoftmaxRows/SumRows/SumCols/Transpose 的移动小于自身散布。

### 关键陷阱：一个双态的内联决策

上表所有收益都追溯到同一个编译器事件。逐元素包装函数（`tensor.Add`、`tensor.Sub`、`tensor.Hadamard` 等）调用 `broadcastBinary(a, b, 闭包)`（`tensor/ops.go:185`），其内层循环**逐元素间接调用**该闭包。应用画像后，`broadcastBinary` 被标记为热点，内联预算提高，`-gcflags=-m` 显示它整体消失进全部三个包装函数（`Add` 在 `ops.go:272`、`Sub` 在 281、`Hadamard` 在 291）：

```
tensor/ops.go:272:24: inlining call to broadcastBinary
tensor/ops.go:272:24: inlining call to Add.func1
```

每元素一次间接调用被消除：128×128 基准上是 16,384 次调用，31.0 µs → 9.4 µs（约 1.9 → 0.57 ns/元素）。热循环由包装函数算术构成的基准按比例受益：`ChainForwardBackward`（16 层 `Add(Hadamard(v, w), x)`）−20.5%、`DivDenLoop` −7.9%。`nn` 细胞基准的收益小于阶段 16 之前实测的 −7%：融合 LTC 内核（`nn/ltc_fused.go`）把 ODE 展开作为一个 `FusedOp` 节点执行，完全不再经过广播包装函数，因此只有步内未融合的残余部分受益（LTCStep/UnrollBackward −3.6%）；而 remat 一对基准——其重算扫描重建的是普通的逐步子图——保留了更大份额（−6%）。

但这个决策是双态的。我们用*完全相同*的命令、相隔几分钟采集了三份画像，外加三者的合并（`go tool pprof -proto a b c`）：

| 画像 | `broadcastBinary` 被内联？ | 实测效果 |
|---|---|---|
| #1（2 s） | **是**（全部三个包装函数） | 上表 |
| #2（2 s，同一命令） | **是**（全部三个包装函数） | —（未复测） |
| #3（5 s） | 否 | — |
| #1+#2+#3 合并 | 否 | ≈ 0（LTCStep 与 Hadamard 均为基线水平） |

用第二个世界的画像时，局面翻转：编译器把热点预算花在了 `broadcastBinary` *内部*（`-d pgodebug=1` 轨迹显示内联了 `broadcastShapeFresh`（开销 ≈1190）与 `bcastMode`），包装函数处再不出现 `inlining call to broadcastBinary`——这与「`broadcastBinary` 自身开销（独立时 811）在吸收这些被调方后胀过扩宽预算（2000）」的推断一致。落到哪个世界，取决于调用树更深一层处的偶然样本分布——采样运气，不是采集者能控制的。Go 官方文档的警告在此逐字适用："*microbenchmarks are usually bad candidates for PGO profiling*"（微基准通常不是 PGO 画像的好来源）——上表是把基准画像用回同一批基准；来自真实应用的画像可能落在任一世界。

端到端看本库自己的例子：`examples/cfc-sequence` 在有/无 PGO 下都跑 ≈ 0.30 s——玩具负载低于计时分辨率；其 stdout **逐位一致**（`first loss 0.620651 -> final loss 0.029091`）。二进制体积从 2,652,082 增至 2,666,898 字节（+0.6%，即官方文档所说的"slightly larger binaries due to additional function inlining"）。

## 工作流：给自己的二进制采画像

1. **从真实（或足够像的）负载采集 CPU 画像。** 任何 pprof CPU 画像都可以——`runtime/pprof.StartCPUProfile`，或服务用的 `net/http/pprof` 端点 `/debug/pprof/profile?seconds=30`。生产负载最好；Go 官方文档强调代表性比时长更重要。
2. **带着画像构建。** 要么把画像放成 main 包目录下的 `default.pgo`（Go 1.21 起自动拾取），要么显式传入：`go build -pgo=/路径/画像.pprof`。Go 官方建议把 `default.pgo` *提交进仓库*以保证构建可复现——画像有源码稳定性（重命名与改动只会让匹配优雅退化），且跨 GOOS/GOARCH 可移植。
3. **验证所得。** 重跑你自己的 A/B 测量。要检查上面那个逐元素大胜在你的构建里是否触发：`go build -pgo=… -gcflags='-m' ./... 2>&1 | grep 'inlining call to broadcastBinary'`。
4. **定期重新采集**，尤其在大重构之后；陈旧画像只会优雅退化（匹配不上的代码失去额外优化），绝不会编错。

### 在本仓库复现上述数字

```
make pgo-profile                       # 写出 /tmp/lnn.pprof（可用 PGOFILE=… 覆盖）
make bench      > /tmp/base.txt        # 基线
make pgo-bench  > /tmp/pgo.txt         # 同一套基准，应用画像
benchstat /tmp/base.txt /tmp/pgo.txt   # 或直接对比两个文件
```

画像是仓库之外的一次性产物，不会有任何东西被提交。注意上文的双态性：如果你采的画像落在"不内联"的世界，重新采集，或按收益区间的保守端预期。

## 注意事项

- **预期 2–14%，不是 68%。** 逐元素的大幅数字是真实的，但悬于一次内抛硬币；对任意应用而言，Go 官方的全局数字才是诚实的先验。
- **PGO 不能替代算法层面的工作**——它只调整内联与布局，不改变你的代码计算什么。在本库上：分配不变，输出逐位不变。
- **首次构建更贵**（所有包都要对着画像重建；之后有缓存），二进制体积略增。
- 不具代表性的画像不应让程序变慢——[Go 官方文档 FAQ](https://go.dev/doc/pgo)（"Will PGO with an unrepresentative profile make my program slower than no PGO?"）的原话是 "*It should not.*"：它只是优化了错误的（冷）函数，热点部分不应变慢。

## 为什么本仓库不附带 `default.pgo`

- 库没有 main 包；examples 是仅有的 main，而它们的运行时长（≈ 0.3 s）低于 PGO 的计时分辨率。
- 提交画像就是提交一个需要随代码与 Go 工具链演进反复重采的二进制 blob——而仓库里仅有的二进制上的实测收益为零。
- 消费者从*自己*负载的画像中得到的收益严格更大；上面的工作流才是交付物，而不是一个 blob。
