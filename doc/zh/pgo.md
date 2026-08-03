# PGO（profile-guided optimization，画像引导优化）

> [English](../pgo.md) | 中文

**摘要：** Go 的画像引导优化（PGO）能为基于本库构建的二进制带来什么、不能带来什么——附在本仓库基准套件上的实测数字、这些数字共同悬于其上的数个编译器决策，以及给*你自己的*二进制采集画像的完整工作流（画像只能放在你那边：LNN 是库，不附带任何画像）。

**读者对象：** 追求吞吐量的用户；纠结要不要在 `main` 包旁放一个 `default.pgo` 的人。

本文涉及的工具链事实均出自 Go 官方 PGO 文档 <https://go.dev/doc/pgo>。测量命令已封装为 `make pgo-profile` / `make pgo-bench` 目标（见下文）。

## 一句话版本

- PGO 自 Go 1.20 起可用；自 Go 1.21 起，`go build` 会自动拾取 main 包目录下的 `default.pgo`（`-pgo=auto`）；`-pgo=<路径>` 可显式指定画像（[go.dev/doc/pgo](https://go.dev/doc/pgo)）。
- 截至 Go 1.22，Go 官方在一组代表性 Go 程序上报告的提升为**约 2–14%**（同一出处）。对真实应用程序，请以此为预期——而不是下面最大的那个数字。
- 本机实测（Go 1.26.5，Apple M4，v0.6.0 树）：当画像把**全部三个**广播包装函数的内联决策都翻转时，逐元素张量算子 ns/op 下降 **−57 ~ −69%**（`AddBroadcastRow` −69%、`Hadamard` −58%；v0.6.0 重采复测，与融合前数字一致）。在本树上**新采**的画像只翻转 `Hadamard` 一个——`Hadamard` 仍拿 −57%，`Add`/`Sub` 保持 ≈ 0——合并画像则一个都不翻转（处处 ≈ 0）。`nn` 融合核基准在*任何*画像下都只有 **≈ 0 到 −3%**：融合 LTC/CfC 内核几乎不再路由包装函数算术，融合把 PGO 原本给细胞步的收益窗口关掉了大半。效应为真，但*多态*，且不受用户直接控制。
- PGO 绝不改变数值：分配次数逐项不变，`examples/cfc-sequence` 的输出在有/无画像两种构建下逐位一致（已验证，见下）。

## 为什么库本身不附带画像

PGO 是**按 main 包**应用的：画像要么是放在*你的* `package main` 旁边的 `default.pgo`，要么是传给 `go build -pgo=` 的路径。库没有 main 包，附带画像无处可挂。而且这个优化是全程序范围的——按 Go 官方文档的说法，"PGO in Go applies to the entire program … including packages in dependencies. This means that the unique way your application uses a dependency impacts the optimizations applied to that dependency."（PGO 作用于整个程序……包括依赖里的包；你的应用使用某个依赖的独特方式会影响该依赖被如何优化。）真正有用的画像，是从*你的*工作负载里采的——那只有你才有。

## 在本仓库上的实测

环境：`go1.26.5 darwin/arm64`，Apple M4（10 核，16 GB），macOS 26.5.2（Darwin 25.5.0）。下表全部数字来自 v0.6.0 重采，于发布树的 `git archive HEAD` 干净快照上（与并行工作树改动隔离）单次会话内三态交错 **A/B/C/A/B/C**（每轮 `-count=3`，每格 n = 6，带 `-benchmem`）：**A** = 无 PGO，**B** = *旧*的阶段 16 画像（本表原版所用的那份，保留以回答"陈旧画像"问题），**C** = 在本树上以标准命令新采的画像（`go test ./nn -run '^$' -bench 'BenchmarkLTCStep|BenchmarkUnrollBackward' -benchtime=2s -cpuprofile=…`）。报告值为六次逐格运行的**中位数** ns/op——这是对旧版"平均值"的有意改动，因为中位数对上一版表格中的单会话离群点更稳健（如阶段 19 的 LTCStep 行）。分配次数在各态逐项完全一致（见下验证）：

| 基准 | 基线 | 旧画像 | 新画像 | Δ 旧 | Δ 新 | 分配 |
|---|---:|---:|---:|---:|---:|---|
| tensor/AddBroadcastRow | 31,966 | 9,918 | 32,330 | **−69.0 %** | +1.1 %（ns） | 不变 |
| tensor/Hadamard | 21,402 | 9,066 | 9,165 | **−57.6 %** | **−57.2 %** | 不变 |
| autograd/ChainForwardBackward | 416,292 | 319,454 | 416,041 | **−23.3 %** | −0.1 %（ns） | 不变 |
| autograd/DivDenLoop | 298,902 | 289,278 | 301,741 | −3.2 %（ns） | +0.9 %（ns） | 不变 |
| autograd/GatherRowsBackward | 10,220 | 10,344 | 10,400 | +1.2 %（ns） | +1.8 %（ns） | 不变 |
| nn/UnrollPeakMemory512 | 65,169,079 | 63,463,448 | 63,760,520 | −2.6 %（ns） | −2.2 %（ns） | 不变 |
| nn/UnrollRematCfC | 884,883 | 874,745 | 882,294 | −1.1 %（ns） | −0.3 %（ns） | 不变 |
| nn/UnrollRemat | 3,391,914 | 3,306,071 | 3,338,338 | −2.5 %（t = 2.7） | −1.6 %（ns） | 不变 |
| nn/UnrollRematPeakMemory512 | 167,316,423 | 160,131,072 | 160,577,974 | −4.3 %（ns） | −4.0 %（ns） | 不变 |
| nn/LTCStep | 82,980 | 83,015 | 80,632 | +0.0 %（ns） | −2.8 %（ns） | 不变 |
| nn/UnrollBackward | 1,361,163 | 1,333,687 | 1,318,196 | −2.0 %（ns） | −3.2 %（t = 3.2） | 不变 |
| tensor/MatMul64 | 77,739 | 79,414 | 78,792 | +2.2 %（ns） | +1.4 %（ns） | 不变 |
| tensor/MatMul128 | 665,644 | 662,934 | 667,760 | −0.4 %（t = 2.2） | +0.3 %（ns） | 不变 |
| tensor/SumCols | 7,650 | 7,684 | 7,758 | +0.4 %（ns） | +1.4 %（ns） | 不变 |
| nn/CfCStep | 36,309 | 36,162 | 35,734 | −0.4 %（ns） | −1.6 %（t = 2.5） | 不变 |
| tensor/SoftmaxRows | 46,322 | 46,661 | 46,041 | +0.7 %（ns） | −0.6 %（ns） | 不变 |
| tensor/SumRows | 7,831 | 7,820 | 7,836 | −0.1 %（ns） | +0.1 %（ns） | 不变 |
| tensor/Transpose | 10,674 | 11,042 | 10,734 | +3.5 %（ns） | +0.6 %（ns） | 不变 |

Welch t 统计量（n = 6 对 6）。**旧**画像下三个逐元素大移动全部显著：AddBroadcastRow 39.6、Hadamard 33.5、ChainForwardBackward 20.1（各 p < 0.001）；UnrollRemat 2.7 与 MatMul128 2.2 为边缘。**新**画像下 `Hadamard` 是唯一大移动（t = 33.1，p < 0.001）；UnrollBackward（t = 3.2，p < 0.01）与 CfCStep（t = 2.5）是小幅边缘到显著的正收益。其余每行移动都小于自身散布——包括两种画像下的 LTCStep（t = 0.0 与 t = 1.1）。一处保留：`ChainForwardBackward` 在新画像下呈**双态**——上表该列为 ≈ 0（表格背后的那份样本），但同一命令另采的一份画像在第二会话给出了完整 −23%。该行收益跟随比包装函数集更深的调用路径，故跨新画像样本不稳定（见下方「三世界」小节）。

**陈旧画像问题，以重采作答。** 旧画像对 tensor/autograd 各行的效果不变：仍翻转同样的三个包装函数、仍给出 −57 ~ −69%——这正是 Go FAQ 优雅退化承诺对"代码未变"的预言。在 `nn` 各行上旧画像如今零代价：LTCStep 为 +0.0%（阶段 19 的 +13.5% 偏劣趋势**未**复现，宜读作单会话离群点而非陈旧画像效应），每个融合核行都是 ≈ 0 到 −3%。在本树上*新采*的画像落在部分内联世界——`Hadamard` 是、`Add`/`Sub` 否（机制见下）——所以新消费者从新画像能拿到的逐元素收益是 `Hadamard` −57% 而非完整 −57 ~ −69%，`nn` 各行则保持 ≈ 0 到 −3%。

### 关键陷阱：三态的内联决策

上表所有收益都追溯到同一个编译器事件。逐元素包装函数（`tensor.Add`、`tensor.Sub`、`tensor.Hadamard` 等）调用 `broadcastBinary(a, b, 闭包)`（`tensor/ops.go:195`），其内层循环**逐元素间接调用**该闭包。当画像把包装函数标为热点、内联预算提高后，`-gcflags=-m` 显示 `broadcastBinary` 整体消失进该包装函数（`Add` 在 `ops.go:291`、`Sub` 在 300、`Hadamard` 在 310）：

```
tensor/ops.go:291:24: inlining call to broadcastBinary
tensor/ops.go:291:24: inlining call to Add.func1
```

每元素一次间接调用被消除：128×128 基准上是 16,384 次调用，31.0 µs → 9.9 µs（约 1.95 → 0.61 ns/元素）。热循环由包装函数算术构成的基准按比例受益：`ChainForwardBackward`（16 层 `Add(Hadamard(v, w), x)`）−23.3%、`DivDenLoop` −3.2%（ns）。`nn` 细胞基准几乎无利可图：融合 LTC 内核（`nn/ltc_fused.go`）与融合 CfC 步把 ODE 展开作为单个 `FusedOp` 节点执行，完全不经广播包装函数，阶段 19 又把感知路径内化——因此步内核在*任何*画像下都 ≈ 0（LTCStep +0.0%/−2.8%、CfCStep −0.4%/−1.6%），而仍构建普通逐步子图的 unroll/remat 内核只保留小幅 −2 ~ −3%。

这个决策不是单个开关，而是三个相互关联的开关。旧阶段 16 画像把**全部三个包装函数**标热，把 `broadcastBinary`（开销 834）内联进 Add/Sub/Hadamard（`ops.go:291/300/310`）。本树上的新画像只把 **`Hadamard` 标热而非 `Add`/`Sub`**——采集画像的 nn 基准构建的是 Hadamard 密集的步图，已不再路由广播包装函数。进入「仅 Hadamard」世界的一条路径——本表新画像列背后那份样本所观察到的：`broadcastBinary` 吸收其热点被调方（`New`、`bcastMode`、`IsScalar`，据 `-d pgodebug=1` 轨迹），开销从 834 胀到 956，于是只塞得进热点包装函数的预算。另一份新采集画像在开销仍为 834 的情况下也到达同一世界，故开销膨胀只是路径之一、而非门槛。相隔几分钟、用*完全相同*命令采集的四份画像外加合并：

| 画像 | `broadcastBinary` 被内联？ | 实测效果 |
|---|---|---|
| 旧（阶段 16） | **是**（全部三个包装函数） | 表格 Δ 旧 列 |
| 新 #1（2 s） | **是**（仅 Hadamard） | 表格 Δ 新 列 |
| 新 #2（2 s，同一命令） | **是**（Add + Hadamard） | —（未复测） |
| 新 #3（2 s） | **是**（仅 Hadamard） | — |
| 新 #1+#2+#3 合并 | 否 | ≈ 0（AddBroadcastRow、Hadamard、SoftmaxRows 均为基线水平） |

因此在 v0.6.0 树上，新画像稳定给出 `Hadamard` 收益（−57%），有时也带上 `Add`，而合并画像落在不内联世界（≈ 0）。落到哪几个包装函数，取决于调用树更深一层处的偶然样本分布——采样运气，不是采集者能控制的。「三世界」框架描述的是 **tensor** 各行的故事；唯一的 `autograd` 大移动（`ChainForwardBackward`）不随包装函数集走——只内联了 `Hadamard` 的画像在第二会话仍给该行完整 −23%，故其新画像下的收益依赖样本（它的收益跟随画像碰巧标热的一条更深调用路径）。Go 官方文档的警告在此逐字适用："*microbenchmarks are usually bad candidates for PGO profiling*"（微基准通常不是 PGO 画像的好来源）——上表是把基准画像用回同一批基准；来自真实应用的画像可能落在任一世界。

端到端看本库自己的例子：`examples/cfc-sequence` 在每种状态下都跑 ≈ 0.14 s——玩具负载低于计时分辨率；其 stdout 在无 PGO、旧画像、新画像下**逐位一致**（sha256 相同；`first loss 0.620651 -> final loss 0.029091`）。二进制体积：无 PGO 2,718,962 字节，旧画像 2,733,906（+0.5%，即官方文档所说的"slightly larger binaries due to additional function inlining"），新画像 2,717,602（−0.05%，基本不变——部分世界几乎不加代码）。

## 工作流：给自己的二进制采画像

1. **从真实（或足够像的）负载采集 CPU 画像。** 任何 pprof CPU 画像都可以——`runtime/pprof.StartCPUProfile`，或服务用的 `net/http/pprof` 端点 `/debug/pprof/profile?seconds=30`。生产负载最好；Go 官方文档强调代表性比时长更重要。
2. **带着画像构建。** 要么把画像放成 main 包目录下的 `default.pgo`（Go 1.21 起自动拾取），要么显式传入：`go build -pgo=/路径/画像.pprof`。Go 官方建议把 `default.pgo` *提交进仓库*以保证构建可复现——画像有源码稳定性（重命名与改动只会让匹配优雅退化），且跨 GOOS/GOARCH 可移植。
3. **验证所得。** 重跑你自己的 A/B 测量。要检查逐元素收益在你的构建里是否触发：`go build -pgo=… -gcflags='-m' ./... 2>&1 | grep 'inlining call to broadcastBinary'`。`Hadamard` 调用点（`ops.go:310`）命中即 −57% 所需；`Add`/`Sub`（291/300）命中则补上其余。把 grep 命中当作**下界**：autograd 链式行可能在无任何 `Add`/`Sub` 命中时仍有收益（其收益跟随更深调用路径），所以仅 Hadamard 命中不代表整条 autograd 线持平。若一无所获，你的画像落在不内联世界——重新采集，或按收益区间的保守端预期。
4. **定期重新采集**，尤其在大重构之后；陈旧画像只会优雅退化（匹配不上的代码失去额外优化），绝不会编错。

### 在本仓库复现上述数字

```
make pgo-profile                       # 写出 /tmp/lnn.pprof（可用 PGOFILE=… 覆盖）
make bench      > /tmp/base.txt        # 基线
make pgo-bench  > /tmp/pgo.txt         # 同一套基准，应用画像
benchstat /tmp/base.txt /tmp/pgo.txt   # 或直接对比两个文件
```

画像是仓库之外的一次性产物，不会有任何东西被提交。注意上文的三态行为：在本树上新采的画像落在"仅 Hadamard"世界（表格的 Δ 新 列）；若你的画像落在不内联世界，重新采集，或按收益区间的保守端预期。

## 注意事项

- **预期 2–14%，不是 68%。** 逐元素的大幅数字是真实的，但悬于几次相互关联的内联抛硬币；在本树上，新画像稳定命中 `Hadamard` 那枚，但不总能命中其余。对任意应用而言，Go 官方的全局数字才是诚实的先验。
- **PGO 不能替代算法层面的工作**——它只调整内联与布局，不改变你的代码计算什么。在本库上：分配不变，输出逐位不变。
- **首次构建更贵**（所有包都要对着画像重建；之后有缓存），二进制体积略增。
- 不具代表性的画像不应让程序变慢——[Go 官方文档 FAQ](https://go.dev/doc/pgo)（"Will PGO with an unrepresentative profile make my program slower than no PGO?"）的原话是 "*It should not.*"：它只是优化了错误的（冷）函数，热点部分不应变慢。

## 为什么本仓库不附带 `default.pgo`

- 库没有 main 包；examples 是仅有的 main，而它们的运行时长（≈ 0.14 s）低于 PGO 的计时分辨率。
- 提交画像就是提交一个需要随代码与 Go 工具链演进反复重采的二进制 blob——而仓库里仅有的二进制上的实测收益为零。
- 消费者从*自己*负载的画像中得到的收益严格更大；上面的工作流才是交付物，而不是一个 blob。
- 若你的应用热循环是逐元素张量算术，PGO 值得一试（`Hadamard` 收益对新画像稳健）；若是融合 `nn` 细胞，预期 ≈ 0 到 −3%，靠融合本身而非 PGO。升级工具链或大重构后应重新采样。
