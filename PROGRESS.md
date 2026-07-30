# LNN 项目实施进度文档

> 主控（orchestrator）维护 · 每阶段结束同步更新
> 项目：`lnn` — 纯 Go 数值计算 / Liquid Neural Network 库（约 1,723 行，3 个包）
> 建立日期：2026-07-29

## 总览

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 并行项目分析（4 路 agent） | ✅ 完成 |
| 2 | 汇总分析 → 产出 PLAN.md 规划 | ✅ 完成（见 PLAN.md） |
| 3 | 按规划实施（目录整顿 + 缺陷修复 + 补测试） | ✅ 完成 |
| 4 | 红队复审 + 全量验证 | ✅ 完成（裁决：生产就绪，置信度 ~90%） |
| 5 | 技术债清扫 + 工程成熟度（Benchmark/CI/双语文档） | ✅ 完成 |
| 6 | 双轨扩展：特性（optimizer/CfC）+ 性能（热路径向量化） | ✅ 完成 |
| 7 | 双轨再进：序列化特性 + autograd 深改 + 发布 | ✅ 完成 |
| 8 | v0.2 双轨：融合反向 + 序列化版本化 → 覆盖率收复 → 红队总扫 → v0.2.0 | ✅ 完成（v0.2.0 已发布；CI 逮住黄金测试跨平台脆弱 → 8e 修复中 → v0.2.1） |

## 阶段 1：并行分析（已完成）

派发时间：2026-07-29

| Agent | 职责 | 范围 | 状态 |
|---|---|---|---|
| 核心分析 A | 数据结构 / 算子 / 梯度正确性 | tensor/ + autograd/ | ✅ 已回报 |
| 核心分析 B | LTC 语义 / wiring / module 设计 | nn/ | ✅ 已回报 |
| 红队审计 | 对抗性漏洞挖掘（panic / 数值 / 并发 / 供应链），含实际复现 | 全仓库 | ✅ 已回报 |
| 工程健康度 | build / vet / gofmt / test / 覆盖率 / 工程设施盘点 | 全仓库 | ✅ 已回报 |

### 产出摘要

**工程健康度（已完成）** — 工程成熟度评级 **2/5**：
- ❌ `nn/` 包**编译失败**：`nn/ltc.go:131` 感官突触路径将整矩阵 `*autograd.Variable` 直接传给要求 `[]*autograd.Variable` 的 `synapses()`（循环路径写法正确，感官路径漏了按行切片）→ 连带 `go vet`、`go test` 在 nn 上被阻断
- ✅ tensor / autograd 编译干净、vet 零告警、测试通过；覆盖率 **autograd 97.6% / tensor 85.7%**
- ❌ `nn/` 零测试文件、零覆盖率
- ❌ gofmt 未格式化：`nn/ltc.go`、`tensor/ops.go`、`tensor/tensor_test.go`
- ✅ 零第三方依赖，`go mod verify` 通过，供应链攻击面为零
- ❌ 仓库基建几乎空白：**非 Git 仓库**，无 README / LICENSE / CI / Makefile / .gitignore / examples / Benchmark

**核心分析 A（tensor + autograd，已完成）** — 总评：数值内核正确、梯度测试扎实，但形状约定不统一、addGrad 无校验、图展开规模是三大隐患：
- 🔴 `autograd/variable.go:45-53`：`addGrad` 只比对 `len(Data)`、**不校验形状**，`[1,6]` 与 `[2,3]` 会静默相加 → 上游形状 bug 被无声吞掉，产出错误梯度
- 🟡 形状约定不对称：`broadcastBinary` 把 1D 结果强升为 `[1,n]`；`SumCols→[m]` 而 `SumRows→[1,n]`；`ops.go:97-99` 标量分支为死代码
- 🟡 `Pow` 反向在 `p==0 且 x==0` 时产生 `0×Inf=NaN`；`SoftmaxRows/LogSoftmaxRows` 对空行 `row[0]` 原生 panic；`MeanAll` 空张量返回 NaN；MatMul 跳零优化吞掉 `0×NaN`
- 🟡 `Backward` 拓扑排序用**递归**，LTC 按时间步展开后图深可达数万节点，存在栈风险
- 🟡 测试盲区：手动 seed Grad、Randn 奇数分支、panic 契约（MatMul/SliceCol/broadcast/Stack/FromRows/At-Set 越界）、`[m,1]` 列广播 Sub 反向均未覆盖
- 🟢 组织问题：`ops.go` 五职责挤一文件（建议拆 linalg/broadcast/activate/reduce）；`Stack` 是产出 3D 却无人消费的孤立 API；nn 四处穿透 tensor 封装（手改 `Data.Data[i]`、手工复现广播判定）；LTC 每步 `SliceRow` 全行拷贝 + 递归反向 = 主要可扩展性瓶颈

**核心分析 B（nn 包，已完成）** — 总评：ODE 数学语义忠实于 Hasani et al. (2021) / ncps 参考实现（半隐式 Euler、参数区间、softplus 约束、数值安全均对齐 ✅），但工程侧亟待修复：
- 🔴 `nn/ltc.go:131` 编译错误确认且**比表面更深**：`synapses()` 内为"传整矩阵"约定写的形状分支是死代码——即便把调用包成单元素 slice 绕过类型错，`ltc.go:170` 的 `muRows[i]` 在 `i≥1` 仍会越界。**LTC 前向从未被执行过**；修法应统一为「接收完整矩阵、内部按行 SliceRow」，删除 recurrent 侧的预切数组
- 🔴 `erev/sErev`（反转电位）被误列入 `Parameters()` 成为可训练参数（`ltc.go:102-103`）——论文语义是固定 ±1 常量，训练会改写突触兴奋/抑制极性，LTC 退化为一般可塑网络
- 🔴 nn 包零测试（再次确认）；全仓无任何 `"lnn/nn"` 调用方，无端到端兜底
- 🟡 文档虚假承诺：`module.go` 宣称的 CfC cell / RNN wrapper / optimizer 均不存在；`Linear.Forward` 与 `LTC.Step` 命名不一，缺 `Cell` 接口与通用 `Unroll`
- 🟡 图规模 O(units²·unfolds) 爆炸；掩码在热路径每突触每子步 `Const` 入图白积梯度（应构造期预掩码）；`RandomSparse` 不校验概率值域、不保证神经元连通性；Wiring 公开构造器允许空掩码
- 🟢 `inW/outW` 恒 1/0 与参考的 U(0.9,1.1)/U(-0.1,0.1) 不对称（可训练故无害）；掩码字段导出可被外部静默篡改；`Div` 借 `Pow(-1)` 在 den≈eps 时梯度达 ~1e16，建议闭式实现；Module 接口过薄（无 Save/Load、无 examples）

**红队审计（已完成，沙箱实测复现）** — 裁决：**当前不可用于生产**。共 13 项发现（1 Critical / 4 High / 5 Medium / 3 Low），均已实际 PoC：
- V-01 Critical：nn 编译失败（与前述一致）
- V-02/V-03 High：`LTC.Step` 对 `ts=1e-40`（float32 溢出）与 `ts=NaN`（绕过 `ts<=0` 校验）静默输出全 NaN——LTC 的标志性变步长场景即是触发条件
- V-04 High：共享参数并发 `Backward` 即数据竞争（`-race` 立即报警），`addGrad` 的 check-then-write 可丢失整块梯度
- V-05 High：`Size()` 整数溢出 → `FromData([],1<<62,4)` 合法返回"2⁶² 行 0 字节"幽灵张量，后续算子非 panic 即死循环
- V-06～V-10 Medium：`New(-2,-3)` 造无合法索引张量；0 列张量触发 Softmax panic；`GatherRows` 按引用捕获 idx → 前向后改 idx 静默腐蚀梯度（实测 grad 错位）；同图二次 `Backward` 梯度 3 倍超线性累加（实测）；`NewLTC` 形状校验本身分配 inDim×units 张量 → 大维度 OOM
- V-11～V-13 Low：除零/溢出无护栏（`Pow(x,0)` 在 x=0 处 NaN 实测）；1D⊕1D→`[1,n]` 与标量叶梯度形状漂移；`Uniform(3,-2)` 静默镜像、Randn 尾部硬截断 7.43σ、wiring 掩码可被构造后篡改
- ✅ 正面：供应链攻击面为零；gradcheck 基础扎实（常规路径构造不出错误梯度）；**10 万层深图 Backward 无栈溢出（推翻核心分析 A 的递归栈风险判断 → 降级为非问题）**；sigmoid/softplus 数值稳定实现正确；随机数可复现

## 阶段 2：规划（已完成）

- 产出 **PLAN.md**：目标目录结构（增量整顿，不做破坏性重排）、P0/P1/P2 工作项共 19 条（每条映射审计依据与验收标准）、红队误报降级留档、执行编排、全局 DoD
- 建立 **Git 基线** `87ccf77`（实施前快照，含 .gitignore），全部改动可回滚

## 阶段 3：实施（进行中）

派发时间：2026-07-29

| Agent | 职责 | 状态 |
|---|---|---|
| 实施 Agent-T | tensor + autograd 修复（P0-3/4/5、P1-2/3/4、P2-3）+ 自触文件 gofmt | ✅ 已回报 |
| 实施 Agent-I | 仓库基建（README / LICENSE / Makefile），不碰 .go | ✅ 已回报 |
| 实施 Agent-N | nn 修复（P0-1/2、P1-1/5/6、P2-1/2/4/6）——等 3a 完成后派发 | ✅ 已回报 |

### 实施回报摘要

**Agent-I（基建，已完成）**：新增 `README.md`（206 行，英文）/ `LICENSE`（MIT）/ `Makefile`（fmt/vet/test/cover/build/all，tab 缩进已自检）。README 最小示例在 /tmp 独立模块**实测跑通**（loss 1.399→0.000，恢复 w≈2.0000、b≈1.0000）；如实披露 nn 编译失败/CfC 未实现/单线程契约（援引 V-04）。附报告 6 处 API 与文档不一致（均已在既有工作项覆盖，无新增工作项）。

**Agent-T（tensor+autograd，已完成）**：11 项工作全部落地，每项附回归测试：
- P0-3 `addGrad` 形状断言（panic 信息含双形状）；P0-4 `GatherRows` idx 入口拷贝（红队错位梯度 [0 1 1 0] → 正确 [1 0 0 1] 回归）；P0-5 `Backward` 后非叶节点 Grad 置 nil（同图二次反向从红队 3 倍超线性 → 精确 2 倍线性叠加）
- P1-2 `Size()` 用 `bits.Mul64` 溢出检查 + `New` 负维度校验（幽灵张量 PoC 转回归）；P1-3 0 列 Softmax 返回空结果、`MeanAll` 空张量显式 panic；P1-4 `Pow` p==0 导数特判为 0
- P2-3 盲区清零：panic 契约表驱动 8 子测试、Randn 奇数分支+方差、手动 seed Grad 路径、Sub 列广播 gradcheck；`Uniform(lo>hi)` 保留镜像行为并文档化（向后兼容决策）
- 验证：`gofmt -l` 空、`go vet` 零告警、`go test -race` 全绿；覆盖率 **tensor 85.7%→89.5%**、**autograd 97.6%→97.7%**；范围自律（未触碰 nn/ 与文档）

**Agent-N（nn 大修 + examples，已完成）**：10 项工作全部落地，nn 包从"编译失败、前向从未运行"修复为可用状态：
- P0-1 `synapses()` 统一为「接收完整矩阵、内部 SliceRow」，删除 recurrent 预切数组与死分支——**LTC 前向首次真正可运行**（冒烟测试断言有限性）
- P0-2 `Parameters()` 剔除 erev/sErev（13 个训练参数，指针+数值双重断言全 ±1）
- P1-1 ts 校验改 `!(ts > 0)` NaN 感知 panic；`scaledCapacitance` float64 计算 + clamp + 可微软上帽（正常 ts 域按位等价于朴素公式，ODE 代数零改动）——ts=1e-40/1e-300/1e300 输出全有限
- P1-5 校验零分配（直读私有 Shape 字段）；P1-6 wiring 概率/维度校验 + 掩码字段私有化 + 访问器返回**克隆**（真只读，热路径 Row 方法直读免拷贝）
- P2-1/2 新增 `ltc_test.go`（9 测试，含全零掩码→纯泄漏闭式回归 1e-4 吻合、5 步 BPTT 15 参数梯度有限、同种子逐位确定性）与 `wiring_test.go`（5 测试）
- P2-4 新增 `cell.go`（Cell 接口 + Unroll + 编译期断言 `var _ Cell = (*LTC)(nil)`）；module.go 删除 CfC/RNN wrapper 虚假承诺（Unroll 兑现时间展开）、补单线程契约
- P2-6 `examples/ltc-sequence`：有界累加器任务（需跨步记忆），**实测 loss 0.690761 → 0.041996（降 94%）**
- 4 项偏离规格均已论证：ts 签名 float32→float64（对齐 ncps `elapsed_time`）、软上帽（仅 clamp 无法消除 Inf×0=NaN）、克隆访问器（指针访问器仍可经 Data 篡改）、example 侧梯度裁剪（不动库代码）
- README 同步更新：nn 片段换为真实可编译 API，删除 "does not compile/preview" 标注，状态表如实改写

**主控独立 gauntlet（不信任自报，亲自复跑）**：`gofmt -l .` 空 ✅ / `go build ./...` ✅ / `go vet ./...` ✅ / `go test ./... -count=1 -race` 全包 ok ✅ / 覆盖率 autograd 97.7%・**nn 98.3%**（目标 ≥70%）・tensor 89.5% ✅ / example 实测 loss 下降 ✅ → 与 Agent-N 报告**完全一致**

## 阶段 4：红队复审 + 全量验证（进行中）

派发时间：2026-07-29 · 红队复审 agent：要求**自写 PoC 逐项销账 V-01～V-13**（不信仓库内回归测试）、攻击新代码（软上帽梯度连续性、Unroll 边界、克隆访问器别名）、核对文档诚实性与 git 历史、给出生产就绪裁决。

### 复审结论（已完成）

**最终裁决：✅ 达到"可用于生产"门槛**（限定于已文档化范围：单线程、适度规模、float32、合理 ts/lr、调用方负责梯度裁剪）。**置信度 ~90%**。

- **V-01～V-13 全部销账**：红队在 /tmp 独立模块自写 PoC 逐项复验（核对所审代码与 HEAD 逐字节一致）。亮点证据：V-02 加测 `5e-324`（float64 下 6/ts=+Inf 被 clamp 兜住）仍有限；V-09 三次 Backward 梯度 `[1,1]→[2,2]→[3,3]` 精确线性；V-10 `NewLTC(50000,50000,…)` 即刻 panic 且仅 10 allocs/run；V-13 篡改四个访问器返回张量后 LTC 输出逐位不变
- **超纲加分项**：红队自写有限差分 gradcheck 覆盖 **LTC 全部 13 个可训练参数**、3 步 ODE 展开，最大相对误差 5.8e-4——梯度不只是有限，而是**正确**；erev 经一轮 SGD 后数值逐位未变
- **软上帽对抗结论**：ts∈[1e-3,100] sweep 与朴素公式 maxDiff=0（首位按位差异在 ts≈1e-38，远低于任何合理训练域），梯度 C∞ 连续，P1-1 宣称属实
- **残余风险（如实披露，非缺陷）**：V-04 并发 race 依然存在，以"单线程契约"文档化销账（README + module.go 双重声明，红队确认披露充分）；float32 无内置优化器，训练稳定性依赖调用方裁剪
- **新发现 F1～F5**：均 Informational/Low，无阻断项——F1 极负 cm+极小 ts 下 cmT 可为负但输出仍有限（tiny-ts 为 finiteness-only 域，留档）；F2 训练需裁剪（**已采纳**：README Quick start 补裁剪提示）；F3 ts=+Inf 被接受（契约一致，留档）；F4 Randn 7.43σ 截尾（既有接受项）；F5 Unroll 空序列语义（**已采纳**：cell.go 注释补齐）
- 采纳 F2/F5 后主控复跑 gauntlet：gofmt 空 / build / vet / test -race 全绿 / example loss 下降 —— 验证状态无漂移

## 阶段 5：技术债清扫 + 工程成熟度 + 双语文档（进行中）

派发时间：2026-07-29（应用户追加要求，文档按中英双语交付：doc/ 英文 + doc/zh/ 中文镜像 + README_zh.md；godoc 依 Go 惯例保持英文）

| Agent | 职责 | 状态 |
|---|---|---|
| 实施 D1 | Div 闭式单节点（债#2）、LTC 拒绝非有限 ts（F3）、Randn 截尾/Stack 孤立 API 文档标注（债#4/#6） | ✅ 已回报 |
| 实施 D2 | 全库 Benchmark（13 函数）+ make bench + GitHub Actions CI | ✅ 已回报 |
| 实施 D3 | 英文文档体系：3× doc.go + doc/ 六篇指南 + README Documentation 索引 | ✅ 已回报 |
| 实施 D4 | 中文镜像：doc/zh/ + README_zh.md + 中英互链（等 D3 定稿） | ✅ 已回报 |

### 实施回报摘要

**D1（数值技术债，已完成）**：四项清扫全部落地，零行为破坏：
- **Div 闭式化**（autograd/ops.go:171-194）：`Hadamard(a,Pow(b,-1))` 双节点 → 单节点闭式，图节点砍半。一致性经**三层验证**：① `broadcastShape` 全部 12 种形状组合前向+双梯度逐位断言（`TestDivMatchesLegacyComposition`）；② 随机 5 组 gradcheck；③ LTC 端到端 example 首迭代 loss 与留档值 0.690761 **逐位吻合**。b=0 前向 [+Inf,-Inf,NaN] 与旧实现逐位一致；nn 内唯一调用点（ltc.go:170）分母恒 ≥ eps，b=0 不可达。2 项偏离均论证（db 求值次序对齐旧 Pow 链以保证 float32 逐位一致；前向复用 `Pow(b.Data,-1)` 走标准库 1/x 快路径排除舍入差异）
- **F3**：`LTC.Step` 校验加 `math.IsInf`，±Inf 均 panic（+Inf 是旧版漏网点，红队 F3 销账）；既有 finiteness 回归不受影响
- **债#4/#6 文档化**：Randn doc comment 注明 7.43σ 硬截尾与"仅适用初始化"边界；Stack 标注 Experimental（3D 产出无算子支持，为 API 兼容保留）——均为纯注释，代码零改动
- 验证：build/vet/test-race 全绿；覆盖率 autograd 97.7%→**97.8%**，nn 98.3%，tensor 89.5%

**D3（英文文档体系，已完成）**：`tensor/autograd/nn` 三个 `doc.go` godoc 门面（79/75/92 行）+ `doc/` 六篇指南（architecture 168 行：三层数据流图、无 stride 设计取舍、图即内存模型；shapes-and-broadcasting 127 行：广播 9 case 全表 + 诚实标注不对称约定；training 220 行：四步回路 + 三纪律 + 裁剪/动量实测范例 + 7 项发散清单；**ltc 218 行：ODE 各项↔ltc.go 行号逐项对照、半隐式 Euler 代数推导、15 行参数表、ts 四档有限性域、ncps 对应表**；pitfalls 188 行：红队发现转用户视角 9 条；README 索引 27 行）+ README 纯追加 Documentation 小节（+16 行）。**全部示例 /tmp 实测**（README quickstart 输出逐行一致、doc.go 三片段、广播/归约全表逐条、二次 Backward [2 4 6]→[4 8 12]、`-race` 并发范式无竞争）；主动与 D1 并行改动实时对齐（Div 行号核至 171-194、+Inf 拒绝已验证）。遗留：三份源码内旧包注释与 doc.go 并存（go 1.26 下 doc.go 胜出，用户无感，未来可清理）

**D2（Benchmark + CI，已完成）**：新增 `tensor/bench_test.go`（8：MatMul64/128、SoftmaxRows、广播 Add、Hadamard、SumRows/SumCols、Transpose）/ `autograd/bench_test.go`（3：16 层链前反向、GatherRows 反向、Div 密集回路）/ `nn/bench_test.go`（2：LTCStep、Unroll+Backward 复刻 example）——13 项实测全通；`make bench`（BENCH 变量可过滤，不并入 all）；`.github/workflows/ci.yml` 7 步（gofmt 门禁/vet/build/test-race/example 冒烟），YAML 经 python yaml 解析校验。**首次量化性能债**：`LTCStep` 7372 allocs/op（~502KB）、`UnrollBackward` 120452 allocs/op（~12MB）——O(units²·unfolds) 图规模问题有了度量尺子；`BenchmarkDivDenLoop`（807 allocs/op）留作 D1 Div 闭式化的回归基线。范围自律确认（未触碰他人文件）。

### 主控前置验证（D1/D2/D3 合并态，不等 D4）

- **Div 闭式化收益量化**（重跑 `BenchmarkDivDenLoop`，-count=3 稳定）：807 allocs/op → **753 allocs/op（−6.7%，每 Div 省 6 次分配）**；750,186 B/op → 712,043 B/op（−5.1%）；454,680 ns/op → ~404,500 ns/op（**−11%**）。留档债 #2 销账且有度量证据
- **全量 gauntlet**：`gofmt -l .` 空 / build / vet / `go test ./... -race` 全包绿 / example loss 0.690761→0.041996 **与留档值逐位一致**（Div 替换未扰动训练行为）

**D3+D4（双语文档体系，已完成）**：
- 英文（D3）：3× `doc.go` godoc 门面 + `doc/` 六篇指南（architecture/shapes-and-broadcasting/training/**ltc**（论文↔代码逐项对照）/pitfalls/索引）+ README Documentation 小节；全部示例 /tmp 实测
- 中文（D4）：`README_zh.md`（202 行）+ `doc/zh/` 六篇镜像（共 931 行），篇首中英互链，术语表 13 词条首现括注英文；结构自检标题数/表格行数逐篇一一对应；全部示例（含 `-race` 并发范式）实测通过
- **D4 回源核对反查出英文版 4 处与代码漂移，已双语修正**：①training.md 裁剪实测值不可复现（16.46/0.003038 → 可复现的 2.50/0.019975）；②shapes 文档"标量×标量→[1]"不精确（实为 [1,1]，仅 0 维操作数得 [1] 且混用顺序敏感）；③ltc.md 位级偏差起点 1e-38 → 1e-37–1e-38（含钳制阈值 1.8e-38）；④README 覆盖率 ~86% → ~90%。"不许纯机翻"纪律兑现为一次额外的文档审计
- 终检 gauntlet 全绿；godoc 依 Go 惯例保持英文（中文文档已注明）

**阶段 5 关闭**。留档表更新：债#2（Div）、F3（+Inf ts）已销账；#4/#6 已文档化；残余项见下表。

## 阶段 6：双轨扩展（进行中）

用户选定双轨并行。派发时间：2026-07-30

| Agent | 轨道 | 职责 | 状态 |
|---|---|---|---|
| 实施 F-A | 特性 | `optimizer/` 新包：SGD/Momentum/Adam，显式 struct，状态按参数指针隔离 | ✅ 已回报 |
| 实施 F-B | 特性 | `nn/cfc.go`：CfC 闭式连续时间细胞（Hasani 2022），取证 ncps 参考实现，向量化自包含 | ✅ 已回报 |
| 实施 P-A | 性能 | `nn/ltc.go` synapses 向量化 + 掩码出热路径；门禁：allocs −50% 且三重正确性验证 | ✅ 已回报 |
| 红队验证 | 验证 | CfC 论文符合度 + optimizer 更新式 + P-A 等价性（等 6a 落地） | ✅ 3/3 全部回报（三裁决皆 ✅） |
| 文档同步 | 双语 | doc/ 与 doc/zh/ 全量同步 + README 路线图改写（等红队后，代码定型） | ✅ 已回报（8 工单全清，20 文件双语同构） |
| 实施 F-C | 特性 | `serialize/` 二进制流包 + `nn/save.go`（LTC/CfC/Linear Save/Load，先校验后分配） | ✅ 已回报 |
| 实施 F-D | 特性 | `examples/cfc-sequence`（CfC + optimizer 范式） | ✅ 已回报（loss 0.621→0.029，与 doc/cfc.md 六打印点逐值吻合） |
| 实施 P-B | 性能 | autograd 反向分配深改（addGrad 去克隆 / broadcast 去闭包），零数值变化逐位门禁 | ✅ 已回报（−50.55%，门禁两度拦截真实缺陷） |
| 红队·序列化 | 验证 | 变异模糊 7,500 体 + 资源耗尽 + 语义攻击 + 动力学等价 | ✅ 已回报 + **F1-F6 修复已销账**（4GiB→33KiB 等，1,200 变异体 0 panic） |
| 红队·autograd | 验证 | 独立差分 fuzz + FMA 屏障千万对验证 + 别名专项 + 基准复测 | ✅ 已回报 + **F1-F3 修复已销账**（52k 图四类差异全零；自证震出同族 F4-F6 一并修补） |

**F-C（序列化，已完成）**：新建 4 文件（serialize 367+531 行 / nn/save 378+572 行），覆盖率 **serialize 97.4% / nn 99.2%**。格式：`"LNNS"` 魔数 + version=1 + 小端张量流（rank≤8、int64 形状、float32 LE、自定界拒尾字节）；模型流 kind 字节（0=LTC/1=CfC/2=Linear）+ header + 17 张量 blob（掩码×2 + 13 参数 + erev×2，顺序单点定义）。**V-05 纪律制度化**：`bits.Mul64` 溢出安全乘法 + maxElems/maxCount/maxRank 限额 + **先校验后分配**——`TestHostileDimDoesNotAllocate`/`TestHostileCountDoesNotAllocate` 以 `AllocsPerRun` 断言 1<<62 维与 0xFFFFFFFF 计数恶意流全程 <50 次小分配、不 OOM；恶意流十一连（bad magic/version=99/截断×2/尾字节/负维/rank=200/乘积溢出/掩码非二值/layout 偷换 [6]→[2,3]）全部语义化 error。**逐位复现保证**：round-trip 经 `Float32bits` 比对（NaN/−0 自等），加载后细胞 Step 输出 + 梯度与原细胞逐位相等（LTC 原位重建 erev 焙入的 numReduce 指示阵是关键一步）；Load 与 rng 种子无关。26 个测试含 quick.Check 200 轮随机 round-trip、跨 kind 互载 7 例、稀疏 wiring 保留、多步 Unroll 逐位。零反射零依赖、未碰任何既有文件——**README 路线图最后一项功能缺口闭合**

**P-B（autograd 深改，已完成）**：留档 #9 销账。七步结构改造（每步独立可回退、过逐位门禁方进入下一步）：A. `addGrad` 首次贡献所有权移交（Clone 占比 19.8%→1.05%，含逐分支别名穷举——Add a 支 `SumToShapeTake` 直通、**b 支保留克隆**防两叶别名腐蚀、根种子在 `Backward()` 集中克隆/归还保 V-09 线性累加与手设种子完整性）；B. broadcastGetter 去闭包（14.2%→0，步长描述+特化内循环）；C. 一元反向链融合（Sigmoid/Tanh 4→1、Log/Pow 3→1 等，**忠实复刻 [1,n] 升维怪癖而非修正**）；D. MatMul 反向去 Transpose（新增 MatMulTransA/TransB，循环形态逐字同构）；E. 乘积-归约融合（hadamardReduce 直累）；F. 反向闭包→opKind 标签派发（~3,168 allocs/op，21 case 逐句等价）；G. 外积形状新鲜度直接采纳。
- **逐位门禁（唯一正当性来源）**：差分 fuzz 96,000 图 × ~160 万节点（4 协议×8 种子×3,000 图，oracle 提取自 git 历史 1aab2de），前向+全叶梯度严格 `==` 且 `Float32bits` 相等（含 ±0），**0 失败**；**负对照**（篡改 oracle Sub 符号）200 图内 42 处差异即 FAIL——fuzz 非空转；example 11 个打印点与基线 **diff 逐字一致**（0.690761→0.041996）；既有数值断言一字未动全绿；新增 8 个别名专项探测
- **门禁两度拦截真实缺陷并当场修复**：①arm64 FMA 融合漂移——融合循环被 SSA 匹配 FMADD/FNMS（单次舍入）vs 旧两步路径（两次舍入）→ 末位 1 ULP 漂移，以 `float32(float64(a)*float64(b))` 转换屏障修复（精确积 ≤48 位尾数在 float64 精确，舍入与硬件 float32 乘法逐位相同）；②1D 升维形状怪癖未复刻 → `elemwiseGradShape` 忠实复刻
- **基准（−benchtime=100x −count=3）**：`UnrollBackward` 68,688→**33,963 allocs/op（−50.55% ✅ 越过 −50% 门禁）**、B/op −50.1%、ns −24%；ChainForwardBackward −57.7%、DivDenLoop −56.7%、LTCStep −29.0%、GatherRowsBackward −23.5%——**五基准全降零回归**。剩余剖析：tensor.New 64.9%（每节点前向输出+Shape/Data 双分配）为下一阶段候选（受阻于公共 API 禁区，留档#12）；Sigmoid-Hadamard 融合反向需新算子层（留档#13）

**F-D（CfC 示例，已完成）**：`examples/cfc-sequence/main.go`——有界累加器任务，CfC(1→8)+Linear、`optimizer.NewSGD` + 手动范数裁剪组合范式（原地缩放梯度后 Step，数学等价于手搓 scale 写法）、显式 ZeroGrad 纪律；loss **0.620651→0.029091（−95.3%）**，两次运行 diff 为空（确定性），六个打印点与 doc/cfc.md 记录**逐值吻合**（双向锁定）

**序列化加固修复（F1-F6 已销账）**：六项全落地，既有断言**零改动**（git diff -U0 删除行为 0，错误信息包装逐字等价保 substring/errors.Is 契约）：
- **F1**：`reader.floats(n)` 二分——已知长度读端（bytes.Reader 等）保持先校验后单次分配的旧行为逐字等价；**未知长度读端（io.Pipe/net.Conn/gzip.Reader）渐进分配**（初始 min(n,16KiB)，4KiB 块读解码 append，EOF 即 ErrUnexpectedEOF）。实测对照：18 字节流声称 [1<<30] 峰值 **4 GiB → 33 KiB（压缩 12.9 万倍）**；[1<<24] 64 MiB → 33 KiB；合法大张量（5×chunk+11，含 NaN/−0）经真 io.Pipe round-trip 逐位正确
- **F2**：`maxUnfolds=1024` 加载路径限额（插在头部解析段、ReadTensors 之前，blob 不解析即拒）；NewLTC 运行时契约不改（构造面 vs 加载面的不对称已论证入 doc）。实测：unfolds=1<<20 从"接受 + 单次 Step 3.58s"→"**2µs 带值拒绝**"；unfolds=1024 合法流 round-trip 逐位正常
- **F3**：张量指针切片渐进增长，count=maxCount−1 分配 **8 MiB → 416 B**
- **F4**：erev/sErev 逐元素位级 ∈{±1} 校验（拒 NaN/±Inf/0/2.5，16 组合全 error，细胞不构造）；**全符号翻图谱可加载**（证明校验认值域而非构造器产物，指示阵按流内极性重建）
- **F5**：LoadParameters "覆写 Data、刻意保留陈旧 Grad、复用前 ZeroGrad" 契约文档化 + 测试钉住（Grad 指针同一性断言）
- **F6**：包注释改为三条精确分项（限额值列明 / 已知长度快路径 / 未知长度按到达字节增长，点名 io.Pipe/net.Conn/gzip.Reader）
- 变异抽测复用红队手法：**1,200 变异体 × 双读端 0 panic**，加固未引入新脆点；五包 `-race` 全绿

**autograd 等价性修复（F1-F3 + 同族 F4-F6 已销账）**：发布阻断解除。
- **F1**：快路产出后校验**乘积真实形状**（方案②——不复制 broadcastShapeFresh 避免第二真相源）：`p := Hadamard(g,x); sameShape(p.Shape,shape) ? p : SumToShapeTake(p,shape)`。热路径仅多 ≤2 整数比较、分配不变。红队最小复现 `[1]` 值 12 逐位恢复；panic 复现场景恢复正常运行（[1]·14）；1D 叶手设种子 [3] 不再升 [1,3]
- **F3 随 F1 自愈**：±0 复现第 4 元素 0x80000000 → 0x00000000（归约恢复 +0 规范化），四元素 Float32bits 逐一相等，固化为测试
- **F2 完全修复（含 NaN 位）**：mul32 非有限操作数走原生乘；**双重实证**——`go tool compile -S` 全包含 FMADD/FNMSUB/FMSUB/FNMUL **合计 0 条**（裸循环对照探针仍发射 FMADDS，证明屏障必要且生效）+ 千万对（全位模式/特殊值/次正规/4 种 NaN 载荷）Float32bits **差异 0/10,000,000**
- **自证震出同族三缺口（红队未覆盖，一并修复）**：F4 融合一元反向对异形手设种子的布局假设（`gradMatchesElemwise` 守卫，不符回退字面旧组合）；F5 hadamardReduce 融合分支 `productCarriesGShape`+`flatSameLayout`+**升维贯通排除**（52k 图第二轮才震出：g[1]⊙x[1]→目标[1,1] 旧链贯通 vs 融合塌缩）；**F6 NaN 符号位**——`-m` 一元负号 FNEGS 翻 NaN 符号 vs 旧链硬件乘传播；坑中坑：**常量 −1 被编译器折叠 Mul32F→Neg32F**，`negOne` 须为包级变量（内存加载折叠不掉，-S 实证）
- **52,000 图 × 三种子差分 fuzz（~50 万节点）**：有限值/形状/panic 有无/NaN 位 **四类差异全部 0**（覆盖桶含 1D 怪癖×广播 500、Hadamard(x,x) 500、扇出 7,899、NaN 梯域 1,579、±0 1,671 等）；负对照 600 图即报 5,893 例差异（门禁非空转）；残留 88-124 例/轮 panic 消息分歧均为 MatMulTransB 改名类表面项（红队已归类）；**自曝两个 harness bug 并修正后三种子交叉**（生成器想象力教训再次应验）
- **性能零回归**：五基准 allocs 逐数持平（UnrollBackward 33,963 不变，2D 热路径快路原样命中）；两 example 逐字；新增 f1_regress_test.go 8 个测试期望值全部对 1aab2de oracle 实测

**红队·autograd 组（已完成）**：自写差分生成器（刻意异于实施方：偏重广播二元 46% + 深扇出 35% + 全怪癖形状池 + NaN 梯域）6,000 图 × 78,696 节点，协议含同图多次反向/手设非叶种子/叶梯度缓冲原位突变/结构别名断言。发现 **254 例差异（4.2%）**，逐一起因定位：
- **F1（Important，必修，发布阻断）**：`hadamardReduce` sameShape 快路对 1D 叶破坏梯度形状契约——旧链 `SumToShape(Hadamard(g,x),target)` 先乘后约，1D 升维怪癖被归约抹平；新快路以"g 形状==目标"短路，但**乘积形状≠g 形状**（`[1]⊙[1]→[1,1]`）。57 例形状违反（值逐元素相等）+ **升级为 panic 回归**：同叶收到 [1] 与 [1,1] 两路贡献时 addGrad 形状断言炸（基线正常运行）——36 例 panic 有无分歧、22 例消息分歧（MatMul vs MatMulTransB 等表面项）。最小复现：`x:=New([3],1); SumAll(Sub(Hadamard(x,x),[2,1])).Backward()` → 旧 x.Grad 形 [1]、新 [1,1]。**与实施方 `elemwiseGradShape` 注释自称的"怪癖忠实复刻"自相矛盾**。实施方 96,000 图零失败系**生成器盲区**（未采样 1D 怪癖×广播 Hadamard；红队生成器 154 图内首发命中，命中率 ~3.3%）
- **F2（Minor）**：138 例 NaN 位漂移（mul32 float64 往返对 NaN 载荷/符号的规范化 ≠ 硬件 float32 传播），两侧皆 NaN、训练语义无影响，但**证伪 PROGRESS 所引"Float32bits 相等"的绝对化宣称**（仅成立于有限值域）
- **F3（Minor）**：1 例 ±0 漂移，与 F1 同源（跳过归约失去 +0 规范化），精确命中"含 ±0"宣称边界
- **F4（正面认证）**：FMA 屏障千万对验证（4M 随机位模式 + 2M 次正规 + 2M 特殊值 + 2M 正态跨 80 数量级）**真实值分歧 0**；`go tool compile -S` 实测 arm64 对裸循环发射 FMADDS——**屏障是承重的，实施方此项完全正确**，"本次审计中质量最高的工程细节"
- **别名安全宣称完全成立（最高风险项经受最强攻击）**：A-E 五个定向 PoC（Add 双叶/三消费者扇入逆拓扑边界/两级直通链/根种子克隆归还/跨实现头对头）双实现通过 + fuzz 内嵌 6,000 图结构别名扫描 **0 命中**
- **性能宣称五项精确属实**：独立复测 UnrollBackward 68,688→33,963（**−50.555%，与宣称逐字吻合**）、ChainForwardBackward −57.65%、DivDenLoop −56.71%、LTCStep −29.01%、GatherRowsBackward −23.53%，ns −38%，三轮稳定零回归
- API 完整性：导出签名零变更；3 个新导出符号（MatMulTransA/B 正当通用算子；`SumToShapeTake` 所有权契约挂公共面为脚枪——保留强文档警示）；既有测试 diff 为空、结构断言未弱化；两 example 输出逐字吻合留档值
- **裁决：既有用法（nn/examples 的 2D 世界）内逐位等价、别名安全、性能属实；但绝对化宣称不成立——F1 为发布前必修项。已派发修复**

**⚠️ 主控更正**：前条 P-B 摘要中"96,000 图严格 == 零失败（含 ±0）"的表述**仅在实施方生成器覆盖域内成立**——红队以异源生成器在 1D 怪癖×广播组合与 NaN 梯域发现 F1-F3。P-B 的门禁机制本身有效（负对照可侦测、确曾拦截 FMA 与升维两缺陷），短板在覆盖域；教训：差分 fuzz 的裁决力不超过生成器的想象力，异源生成器交叉是必需而非可选

**红队·序列化组（已完成）**：7,500 变异体（位翻转/删除/插入/块交换，25% 叠连击）**0 panic / 0 静默错乱**（黑盒 oracle 重序列化核验掩码二值/形状/Step 健全；7 例"ok 垃圾入参"均为参数含 NaN/Inf 的忠实复现）；语义攻击全挡（张量顺序偷换/掩码注入 0.5/−1/2/NaN/跨 kind/未知 kind/version 0·2·99·255/UTF-16/大端伪流）；错误全透传（写端每个截断点、读端多偏移）；round-trip **训练动力学逐位等价**（加载后再训练 3 步的参数轨迹与同步训练锁死、种子无关）。**资源耗尽维度不安全**：F1 Medium——**无 Len() 读端**（网络/管道/gzip）绕过剩余字节守卫，~20 字节截断流逼出 64MiB～**4GiB** 分配（make 在读之前）；F2 Medium——`unfolds` 无上限，1<<20 展开 2.26s、1<<30 外推单次 Step ~38 分钟 CPU 耗尽（CfC 免疫）；F3 Low——count=maxCount−1 强分 8MiB 指针切片；F4 Low——加载不校验 erev∈{±1}（可造 NewLTC 永不能产生的细胞）；F5 Info——LoadParameters 保留陈旧 Grad 未文档化；F6 Info——限额私有且包注释"绝不无界分配"措辞掩盖 F1。**裁决：panic/语义维度安全，资源耗尽维度需补防——已派发修复（发布前置条件）**

**8d 双语文档同步（已完成）**：12 个 .md 文件（+292/−104），五工单全双语落地经回源+/tmp 实测双重核验：
- **F2**：pitfalls §2 补 NaN 对 Momentum/Adam 矩估计**永久毒化**（ZeroGrad 无效——毒在优化器缓冲不在 Grad；须重建优化器，引红队 60 轮实测）；MatMul 跳零补方向性（只测左操作数：0×NaN→0 而 NaN×0→NaN，回源 tensor/ops.go:20 确认）——pitfalls 与 architecture 双处
- **F3**：5 处 erev #10 陈旧"开放问题"→"已完成（阶段 8）"（pitfalls/README×2/cfc.md×2 逐处检索修正）；runBackward 21→**23** case（grep 实证含 opLeaf/opSigmoidHadamard）；ltc.md 示例改 SigmoidHadamard（回源 ltc.go:347 唯一采纳点，cfc.go:279 旧式未误融合已核对）
- **F-RT1**：ltc.md/cfc.md（双版）"逐位等价"宣称补诚实角落——全掩蔽突触后列 ±0 符号位（/tmp 实测 (−1)·(+0)=0x80000000 vs MatMul +0，(±0)²=+0 下游不可观测）；persistence/architecture 回源核对判定**无需**限定（round-trip 同用收缩仍真、复杂度叙述非等价宣称）——不虚改
- **v0.2.0 新内容**：persistence 补 maxUnits/maxInDim=256 限额表 + 量化段（256 MiB 封顶/~320 MiB 峰值/13 字节攻击流 ~550GB→带值 error/load-only 不对称论证/根因见 #14，错误串 /tmp 逐字核对）+ 黄金向量节（-write-golden 门禁 + 三测试角色）；architecture 新增 SigmoidHadamard 小节（三段等价叙事 + 结构核算 68×2=136/396×5=1,980 + aux 双语义）+ 性能刷新 LTCStep 2,306 / UnrollBackward 31,983（本机复测逐字吻合）；README 双版 Status 表覆盖率 100/99.7/100/97.8（go test -cover 实测）+ serialize 行封顶说明 + Roadmap #13/#10 销账 #14 领衔
- **行号终检**：ltc.go（345 行后整体 +2）、cfc.go（drive 重写后 +20~26，12 处引用全刷新）、ops.go Div 793-810/GatherRows 855、tensor.go Stack 170；累计降幅重算 −69%/−73% 保内部自洽
- **回源额外偏差 4 处并修**：cfc.md Algorithm 1 节机制描述陈旧（8a 已重写 drive 为向量块+双 MatMul 收缩）据 cfc.go:246-294 重写；pitfalls "两"项→"一项"（#13 销账后）；architecture aux 注释补 SigmoidHadamard 语义（P-C 遗留闭合）；累计百分比重算
- 收尾：build/test-race 全绿、18 篇 md 零断链、中英 8 对同构、非文档零触碰

**8c 覆盖率收复（已完成，全面超目标）**：以 `go tool cover -func` 实测缺口为唯一选题依据（工单建议中已被既有用例覆盖的项核对后**不凑数重复**），三个新增测试文件 32 个测试（全部值/形状/位模式/panic 消息实质断言，禁凑行式）：**autograd 87.3%→100.0%、tensor 81.7%→99.7%、nn 99.0%→100.0%**（目标 ≥95/≥88/≥99）。回退类测试以 tensor 级旧组合链逐位重建期望（分支"忠实复刻旧链"的存在性契约本身即断言）。
- **变异致死抽样 18 处 / 17 致死 / 1 逃逸（94.4%）**：逃逸项 M17（`if n>1`→`if n>0`）经论证为**等价变异体**——n=1 时 denReduce 退化单位阵，MatMul(flat,I) 与捷径 den=flat **逐位恒等**，任何断言原理上不可杀；其存活反向确证了源码设计声称，且同断言下 M16（den→numReduce）被同一测试立即致死证明监视有效（逃逸非盲区）
- **唯一残余语句论证为真不可达**：broadcastBinary 双常量填充循环体（cols≡1 ⇒ `for j:=1;j<cols` 永不执行，[1,1]×[1,1] 被同形快路径先截）——列明不强凑
- 源码与既有测试**一字未动**（git diff 空）；两 example 逐字；F-RT3 测试命名归一（TestWriterStability→TestGoldenWriterStability，`-run Golden` 可选中冻结三件套，主控补修）

**F1 修复（v0.2.0 阻断项销账，已完成）**：load-only 常量 `maxUnits = maxInDim = 256`（与 maxUnfolds 并列，save.go:82-119，doc comment 含完整推导与红队出处）。**内存模型先与红队实测对账证明精确**：units=512 公式 2·units³·4B+瞬时重建 = 1.61GB vs 红队实测 1,560MB ✓；units=4096 = 549.8GB vs 红队"~550GB 强杀" ✓。取值推导：持久最坏 2·(units³+inDim·units²)·4B = **恰 256 MiB**，加载峰值 +64MiB 瞬时重建 ≈320MiB 封顶；示例/黄金 units≤8，256 给 32× 头room。校验在头部解析段、**blob 解析之前**（LoadLTC save.go:332-337 / LoadCfC:427-432，紧跟 maxUnfolds 检查）；LoadLinear 无指示阵不受影响（已核对）。
- **红队 PoC 场景前后对照**：units=4096 十三字节攻击流从"尝试 ~550GB → 进程 jetsam 强杀"收敛为 **8 allocs / 186 B 的带值 error**（`nn: LTC header has units=4096, exceeding the load limit 256`），压缩 ~30 亿倍
- **门禁工具发现**：`testing.AllocsPerRun` 只数次数不数大小——F1 威胁恰是大小，故新增 `allocBytesPerRun` 字节预算 helper（save_test.go:633），双门禁（次数 ≤50 + 字节 <1MiB）
- **AtLimit 真实触达**：`NewLTC(4,256,nil,1)` round-trip → 13 参数+erev 逐位、Step 输出 Float32bits 逐位（上限不误伤合法使用，0.04s 完成）；回归测试三组（LTC/CfC units × {257,4096,MaxInt32} 三档 + inDim 同构）
- 全绿：五包 -race、黄金向量 units=6 保持、两 example 逐字；仅改 save.go + save_test.go（构造器契约不动，沿用加载面/构造面不对称论证）

**红队·新代码专项（三条独立证据链：现役 API 组合 oracle + git 历史真旧版差分 + 黄金流字节手术，已完成）**：**三项变更逐位等价宣称在敌对值谱下全部成立**——
- SigmoidHadamard：26 组广播组合×敌对值谱（饱和 ±1e6、次正规 1e-45、NaN 双载荷 0x7FC00000/0x7FC0ABCD、±Inf、怪形）前向+反向 dz/dw **Float32bits 零差异**；异形种子回退 19 组值+形状双逐位；panic 消息与旧链逐字相同；共享操作数 NaN 载荷顺序攻击 7 组未兑现宣称的交换律风险；aux 生命周期（二次 Backward 恰 ×2、多消费者图不改写）；采纳点全仓仅 ltc.go:347，cfc.go:279 旧式未被误融合
- 黄金向量：篡改一字节**确实失败**（防空转：末字节/中段/expected.txt token 三种手术全 FAIL，还原复绿）；**手工逐偏移解码 golden_v1_linear.lnns 120B 并按 MatMul 舍入序手算 Forward，6 个 %08x token 逐字吻合**；版本门先于任何分配（v=2 携恶意 count/rank/1<<62 维只在版本门报出，无解析泄漏）；追加式规则与代码无裂缝；-write-golden 默认 false、默认运行 SKIP、testdata shasum 不变
- erev 焙入：真 pre-8a 代码差分 10 组 CfC 配置前向+全 13 参数梯度**逐位零差异**；流内 erev 符号位手术 → 加载后 Step **必变**（指示阵确从流重建）；erev=2.0/0/NaN/±Inf 六值×双路径×双细胞全拒；反射确认字段类型 *tensor.Tensor（死梯度清零）
- **新发现**：F-RT1 低——**±0 符号位角**：全掩蔽突触后列+erev=−1 时旧 Add 链 −0（0x80000000）vs MatMul 收缩 +0，反向零梯度同；(±0)²=+0 使下游不可观测，但"逐位等价"宣称在此角严格为假（机制随 LTC 7b 已发布、8a CfC 继承——8d 文档精确化措辞）；**F-RT2 高（独立确认总扫 F1 且缩窄归属）**：erev 焙入使 **CfC 构造器与 LoadCfC 新增 O(units³) 实体化**——pre-8a cfc.go 对指示阵 **0 引用**，确为 8a 新攻击面（实测 units=256 流 1.3MB → 207MB ≈160×）——已在 F1 修复覆盖内（load-only 上限含 LoadCfC）；F-RT3 信息——`-run Golden` 不匹配 TestWriterStability（命名待 8c 归一）
- **裁决**：等价性 ✅ 可信（唯一例外为实践不可达的 ±0 符号角）；序列化防护非空转、拒绝语义可操作无泄漏；**安全性待 F-RT2（=总扫 F1）修复后放行**

**红队·v0.2.0 全库总扫（新鲜视角，不读既往发现清单，已完成）**：7 类攻击面（52 条文档断言核验 / 15 项数值病理 / 8 项状态别名 / **1,400 定向变异体×4 入口≈6,200 次恶意流调用** / 3 组端到端病理回路 / 跨包组合 / 文档三角裂缝）。
- **F1（High，未披露，阻断 v0.2.0 安全宣称）**：`LoadLTC/LoadCfC` 头部仅校验 `dims≥1` 无上限，而 `sumIndicator/reversalIndicator` 实体化 **[units², units] 指示阵 → O(units³) float32 内存**——ltc.go:310-317 注释只审计了 MatMul 跳零的**算力**没审计**内存**（历史盲区）。PoC 双阶段：合法流 units=512（5.0MB）加载成功但分配 **1,560MB（放大 311×）**；最小攻击流 units=4096（64MB）→ **子进程 signal: killed**（尝试 ~550GB，macOS jetsam 强杀——比 panic 更糟，不可 recover）。**直接击穿三处明示契约**（pitfalls#10/persistence 摘要/serialize 包 doc "敌意流按交付字节比例分配"），且 maxUnfolds=1024 先例证明项目本有此防御范式——units 是被遗漏的孪生面。**裁决：修后放行**（load-only 上限，maxUnfolds 同构，数行）
- F2（低）：NaN 在 Momentum/Adam 状态中**永久毒化**（实测单 NaN 注入→60 轮起全 NaN 永不恢复），pitfalls §2 "该轮剩余"措辞偏轻；architecture.md MatMul 跳零只述 0×NaN→0（实测 NaN×0→NaN，跳过只查左操作数）——措辞方向性不完整
- F3（低）：8a 后 5 处文档陈旧——erev #10 已修但 pitfalls:233/README:262-264/README_zh:208/cfc.md:363-368/zh/cfc.md:262 仍称开放；runBackward 21-case→实际 23（含 opLeaf/opSigmoidHadamard）；ltc.md:72-84 示例为融合前写法
- F4（信息级，不计）：手构不一致张量 String() 裸 panic（违反结构体不变量，契约不覆盖）；另发现遗留空目录 `asym/`（主控清理）
- **正面认证 11 项**：字节流层 6,200 次恶意调用 **0 panic**；模型级语义校验全拒；**训练失败无静默模式（"最重要的负面结果"——未找到任何产出貌似合理错误数字的路径）**；Adam×LTC 500 步 0/500 非有限；ts 契约、梯度正确性（SigmoidHadamard/Div/LogSoftmax/Pow/Softplus/GatherRows gradcheck，MatMulTransA/B 位级一致）、状态隔离（访问器篡改 Step 位级不变、Save/Load 独立性、失败加载原子性）、LoadParameters 语义逐字相符、文档数值宣称**全部逐字实跑吻合**（quick-start 六行/两 example 含中间点/SGD 位级一致）、CfC exprel 阈值差 8.32e-13 与 B⁵/120 理论余项精确吻合、已披露脚枪全守约
- **总评**："文档诚实度罕见高的库"；F1 修复后放行 v0.2.0；根因级修复（稀疏收缩消灭 O(units³) 实体化，构造器层面同存此悬崖）立留档 #14

**F-E（序列化版本化 + erev 死梯度清除，留档 #10 销账，已完成）**：
- **格式冻结四件套**：黄金字节流 `testdata/golden_v1_{ltc,cfc,linear}.lnns`（1607/1603/120 B，固定种子 101/202/303，参数全文档化）+ `.expected.txt` 逐元素 `%08x` 位模式期望（可手工审计）+ 包注释「Format versioning」节（v1 逐字节冻结语义、**追加式演进规则**——新数据只能尾部追加计数张量、kind 注册表只追加不复用、或整体 version=2 升级；**拒绝而非猜测**——高版本报"written by a newer version, update this build"、低版本报"no earlier layout exists"）+ `-write-golden` 门禁式再生成。三测试：加载逐位（Float32bits）+ 写端 `bytes.Equal` 逐字节 + 双读端类一致
- **交叉验证红利**：黄金 LTC 向量在并行 P-C 改动 ltc.go **之前**生成，改动后三测试仍全绿——独立字节级证明 P-C 改动透明
- **erev 焙入（#10 销账）**：复用 ltc.go 的 sumIndicator/reversalIndicator（ltc.go 零改动）建 den/numReduce 指示阵（±1 焙入 numReduce），erev/sErev 字段 `*autograd.Variable → *tensor.Tensor`（rng 抽取位点一字未动）——死梯度从"为零"升级为**结构上不可能**（反射断言字段类型无 Grad 可言）；drive() 逐突触 Hadamard+Add 链 → 块拼 [batch,n·units] 双 MatMul 收缩
- **逐位证据**：cfc-sequence **11 个打印点全部与留档逐字相等**（0.620651→0.029091，含 iter 100=0.041556 = persistence.md 记录值）；全参 gradcheck 8.626e-3 与改前 8.63e-3 同量级同舍入；白盒 oracle（内嵌旧 drive、erev 重包 Var 叶）新旧四输出 + 8 参数梯度 **Float32bits 逐位相等**，且 oracle 确有死梯度（max|∂L/∂erev|=1.631e-1）——对照非空转
- **加载侧回归捕手**：LoadCfC 按流内 erev 重建指示阵（对齐 LoadLTC 先例）；`TestLoadCfCAcceptsFlippedReversalPattern` 断言翻转图谱加载后输出**确实改变**（若忘记重建则纹丝不动——正是此回归的捕手）；cfcTensors 17 张量序不变（黄金依赖）
- 版本错误信息按方向分流补强（保留原前缀供文档平滑过渡）+ 三断言测试；覆盖率 serialize 97.8% / nn 99.0%（微降 0.2pp 系 P-C 新增 ltc.go 代码，8c 收复）

**P-C（Sigmoid-Hadamard 融合反向，留档 #13 销账，已完成）**：新增 `autograd.SigmoidHadamard(z,w)` 单节点（opKind 派发、aux 槽存 sigmoid 输出 s 复用不重算），采纳于 `nn/ltc.go:347` synapsesRows（sensory/recurrent 共同入口）。
- **前向逐位为构造性**（逐字调用同一 tensor.Sigmoid+Hadamard 代码）；**反向意外达成逐位等价**（设计书诚实预期"反向不逐位、用容差门禁"，实际通过舍入位点对齐——`mul32(g⊙w)` 复刻旧链中间张量舍入、外层分组复刻旧 opSigmoid 融合循环——常规 2D 路径 Float32bits 逐位，已固化测试；非 2D/异形种子走**逐字复刻旧双节点**的回退路径，升维怪癖与 panic 契约原样保留）
- 门禁七项全绿：前向 11 组（含饱和/±Inf/NaN/次正规）逐位 + aux==sigmoid(z) 断言；gradcheck 3 种子×8 组合（**初版两例超限经对照诊断为 w≈0 近零梯度 FD 条件数噪声——组合实现同抽取同样超限、解析梯度逐位相同**，调整抽取区间后全绿，容差 2e-2 未放宽、既有断言零改动）；零掩码闭式回归、全参 gradcheck 9.39e-5、BPTT 有限、确定性、既有向量化 oracle 全绿
- **example loss 0.690761→0.041996 逐位不变**（常规路径反向逐位 ⇒ 训练轨迹零漂移）；cfc-sequence 旁证逐字一致
- **基准（同时间窗 A/B，排除整机漂移）**：LTCStep 2,442→**2,306 allocs/op（−5.6%）**；UnrollBackward 33,963→**31,983（−5.8%）**、B −6.2%、ns −3.4%。**结构核算精确闭合**：LTCStep 68 点位×省 2 分配=136=差值 ✓；UnrollBackward 396 点位×省 5=1,980=差值 ✓
- **诚实校准**：收益为个位数百分比——每点位可省的就是"1 图节点+1 反向中间张量"，剩余由 tensor.New 逐节点固定开销主导（留档 #12 公共 API 禁区），此即 #13"收益/脆弱性比"的**实测终答**：三降零回归但结构上限已到
- compile -S 实证：融合循环 **FMADD/FNMSUB/FMSUB/FNMUL 合计 0 条**（正对照裸循环发射 FMADDS，检测灵敏）；并行冲突自适应：nn 包因对方在改 cfc.go 暂时编译失败期间，全部 nn 验证移 /tmp 沙箱（HEAD+仅己方 4 文件）五包 -race 绿
- 遗留：variable.go aux 字段注释仍为 Div 语义（文件范围纪律未改，双语义在 ops.go 注释完整说明——8d 文档工单）

**发布至 GitHub（7c+，已完成）**：仓库 **https://github.com/qorm/LNN（public）**。远端原有用户初始提交（46ab9ff，仅 LICENSE），以 `--allow-unrelated-histories` 合并，LICENSE 冲突按**用户初始提交的既有选择**解决（Copyright (c) 2026 QORM，替换本项目的 LNN Authors 占位）；v0.1.0 标签重定位于合并提交（1eb157a）保证发布快照 LICENSE 与 main 逐字一致。完整历史（基线→阶段3a/3b/4/5/6/7a/7b/7c 全部原子提交）已推送；双语 README 加 CI + Go Reference 徽章。**验证三连**：①GitHub Actions 3 次运行全 `success`（含 v0.1.0 tag 流水线：gofmt 门禁/vet/build/test-race/example 冒烟）；②`go list -m github.com/qorm/LNN@v0.1.0` 经 proxy.golang.org **实测解析成功**——`go get` 全球可用；③仓库可见性 PUBLIC 确认

**7c 双语文档同步（已完成）**：7 工单全部落地。新建 **doc/persistence.md 双语篇**（LNNS 格式规格表 + 六 API + 不可信流安全契约逐条回源 + 训练→保存→加载→续训示例 /tmp 实测逐字入文：60 轮训练→SaveCfC 1859B+WriteParameters 71B→异 seed 加载 Step 逐位相同→续训与无中断同迭代打印逐位吻合→恶意流三连全 error）；architecture.md 双语新增「梯度缓冲区移交非克隆」与「融合反向+FMA 屏障」两节、五基准数字刷新；README 双版加 serialize 行、**删除过时的序列化路线图句**、补 cfc-sequence；pitfalls 路线图表刷新 + 新增§10 持久化安全边界；doc 索引阅读顺序改 training→persistence→ltc→cfc→…；30+ 条 ltc/cfc 行号回源**零偏差**。**回源反查修正 7 处偏差**：覆盖率实测改写——serialize 97.4%→**97.8%**（加固新增测试），**tensor 89.5%→81.7% / autograd 97.8%→87.1%**（深改新增 MatMulTransA/B 跨包代码与防御性回退分支未覆盖，README 如实注明降幅成因不掩饰）；pitfalls 两处旧行号（Div 171-194→725-742、GatherRows 271→791）；6 处"反向闭包"过时措辞；全文档 allocs 旧值刷新（3,440/68,688→2,442/33,963，基线统一 7,360/120,163）；"少于 50 次分配"措辞修正为"不超过 50 次"（与测试断言方向一致）。终检：gauntlet 全绿、18 篇 md 零断链、中英九对同构、非文档文件零触碰

**模块路径迁移（7c+，已完成）**：`module lnn` → **`module github.com/qorm/LNN`**，37 文件 187 行纯路径替换（go.mod + 27 .go 文件 45 对 import + 8 篇文档代码块 + 双版 Installation 重写为 `go get github.com/qorm/LNN@latest` 正式写法、replace 降级为源码备选）。核查：`grep '"lnn/'` 残留 **0**、活文档路径残留 **0**、五包 `-race` 全绿（包名已为 github.com/qorm/LNN/*）、两 example 输出**逐字节不变**、/tmp 下游消费者模拟五包 API 调用通过、外部 URL（ncps/CfC/DOI）零误伤、PLAN/PROGRESS 历史档案保留裸名。6 项判断性取舍留档（错误串 "not an lnn tensor stream"、表头自称、Makefile 注释等名字语境不改）

**用户指示（阶段 7 期间追加）**：①README 致谢 LNN 相关团队——**主控已双语落地**（Hasani/Lechner/Amini/Rus/Grosu 两位论文、mlech26l/ncps、raminmh/CfC、MIT CSAIL/TU Wien/IST Austria/Liquid AI）；②项目发布至 **github.com/qorm/LNN**——`gh auth` 确认 qorm 账号已登录；发布编排纳入 7c+：**模块路径迁移 `lnn` → `github.com/qorm/LNN`**（go.mod + 全仓 import + 双语文档代码块 + godoc 交叉引用，API 破坏性但发布必需，等 7a/7b 代码定型后统一执行）→ Installation 段改 `go get` → 仓库创建/推送/tag/CI 徽章

**文档同步（6c，已完成）**：8 工单全部落地，20 文件（18 改 2 新）双语同构：①7+ 处"无优化器"旧文案清零（全仓 grep 仅剩序列化一项真实未实现）；②training.md 双语新增 optimizer 用法节（Step 契约/梯度累积/超参热改/指针键语义/别名耦合警示）；③**新建 cfc.md 双语篇**（NMI 2022 正确出处 + "liquid cubic 非官方来源"命名警示 + Lemma 1 逐项对照 + exprel 稳定化 + 与 LTC 同 ODE 收敛阶 + 可运行示例实测 loss 0.621→0.029）；④ltc.md 行号表 18 行按重写后源码全量刷新 + 向量化小节 + **ULP 级等价诚实表述**；⑤架构/README 性能数字更新（O(units²)→O(units)）；⑥optimizer/doc.go 别名警示 + sgd.go 信任模型注释（纯注释，git diff 非注释行变更为空）；⑦双语索引与互链（126 条相对链接零断链）；⑧状态表更新（optimizer 行、CfC 行、nn 覆盖率 ~99%）。示例实测 7 组（含 SGD 与手写循环输出逐字一致、CfC vs LTC 收敛阶复测）。**反查新偏差 3 处并修正**：examples/main.go 注释旧文案（主控已补修）、英文版 training.md 裁剪段与中文版漂移（源码 seqLen=12，中文版正确，已统一）、nn 覆盖率 ~98%→~99%

**阶段 6 关闭**。检查点：b7f302c（6a）→ fd14fdf（6c）。

### 红队验证回报（6b）

**CfC 组（✅ 忠实且可信）**：
- **取证（含自纠）**：双版论文 PDF 全文 + ncps/raminmh 两仓完整 tarball + 上游 LTC 论文 + 公网，六路 grep **"liquid cubic" 零命中**——实施方拒采成立；NMI 发表版核验 Theorem 1/Eq.(8)/Lemma 1/Algorithm 1 **俱在**（本组初审据 arXiv v2 曾怀疑引用瑕疵，经发表版交叉验证**自我证伪并撤回**，留档示透明）
- **方程级对照**：源 ODE（arXiv Eq.1 原文）↔ cfc.go 的 `cm·dv/dt=−G(v−A)` 逐项恒等；Lemma 1 闭式解 `v=A+(v₀−A)e^{−κt}` ↔ Step 的 A/κ/F 代数恒等（含"冻结激活=分段常数输入"前提）；Algorithm 1「逐突触编译、允许任意稀疏 WAdj」↔ drive() + wiring 掩码结构同构。裁决：**比官方 pure 模式（MLP 代理闭式解）更贴方程**
- **数值对抗 10/10 全过**：阈值 1e-2 跨越 8001 点扫描跳变 ≤2.98e-8（容差的 1/300）；全参数 gradcheck 零失败（worstAbs 6e-6）；掩码置零突触梯度 9/9 恰为 0.0；极端 ts 8/8 有限（ts=1e-300 ⇒ v 逐位不动、ts=1e300 ⇒ v→A）；dt→0 收敛阶 p≈1.89（O(h²)，优于要求）；**CfC vs LTC 同 ODE 一阶收敛 p≈1.0**（"同一 ODE 两种积分器"宣称成立）；同种子 CfC/LTC 13 参数+erev 逐位同构
- 新发现均 Info：①erev 以 Var 叶入图产生死梯度（Parameters 排除故优化器不可达，微量浪费，留档）；②`v+(A−v)F` 在敌对巨态下抵消丢精度（动力学不可达：A 有界+凸组合 ⇒ 状态永有界）；③平方损失+病理参数溢出毒化反向（损失侧本性，非 cell 缺陷）

**optimizer 组（✅ 正确且干净）**：自撰三套参考实现（f64 教科书式 / f32 同序镜像 / 闭式解），随机梯度流（含 10⁻³–10³ 尺度跳变）对照——SGD 镜像**逐位一致**、Adam 对 f64 Algorithm 1 最大偏差 **1.63e-6**、t=1 闭式 rel=**0 精确**；**鉴别力校验**：错误 t 公式偏离 0.74–1.6%、Eps 位置判别实验（√v̂+ε 偏差 0 / √(v̂+ε) 偏差 0.386）证明测试非空转。状态对抗五场景（指针别名/同指针换 Data/nil-Grad 跳步不推进 t/finalizer 实证 map 钉住无泄漏/双重 Step 精确双应用）全部与文档逐字相符。Adam 在 autograd 手构 Rosenbrock 2D 2000 步 24.2→7.9e-4 无 NaN；SGD 大 lr 发散为干净 ±Inf 零 panic；19 例非法超参全 panic 带值；覆盖率 100% 实测为真。**新发现**：①中/文档——**全库 7+ 处仍宣称"无优化器"**（README×2/training/pitfalls/nn doc.go/module.go），与新包直接矛盾（6c 文档同步的头号工单）；②低——optimizer/doc.go 宜补"别名 Variable 经 Data 共享耦合更新"警示；③低——`NewSGD(+Inf)` 被字面校验放行（信任模型自洽，建议注释）；④低——构造后字段可改非法值（已披露的信任取舍）；⑤Info——手设短 Grad 触发 index panic（autograd 路径不可达）

**性能组（✅ 等价且属实）**：从 git 历史（66d4adf）提取重写前 ltc.go 作独立 oracle，13 组随机化配置差分——前向最大差 **1.79e-7**、全参数+BPTT 梯度最大差 **1.19e-7**（远优于门禁）；掩码置零突触梯度 `==0` 精确断言 352 位 0 违规；指示矩阵白盒核对（含 erev 符号焙入手算例，极性正确）；基准同机复测 **−53.3%/−42.8% 逐字复现**；9 种形状对抗探测 allocs **全部严格更优**无反例；`ltc_test.go` 只增 157 行/删 0 行、导出符号零变更。**唯一保留**：「逐位等价」仅对孤立 `synapses()` 严格成立，整 Step 为 ULP 级等价（eps/常量项结合律重排，~1e-7 良性，非缺陷）——实施报告的措辞应修正为"synapses 逐位、整 Step ULP 级"。

### 实施回报摘要

**F-B（CfC 细胞，已完成）**：`nn/cfc.go` + 11 组测试，nn 包覆盖率 97.7%→**98.7%**，`-race` 绿。
- **取证纠错（对抗式取证，不盲从任务书）**：任务书所给 DOI 经 Crossref 实测指向一篇合成生物学论文——真实出处为 **Nature Machine Intelligence 4, 992–1003 (2022)，DOI 10.1038/s42256-022-00556-7**（arXiv 2106.13898）；取证源含两版论文 PDF 全文 + ncps `cfc_cell.py` + 官方代码 `raminmh/CfC`
- **"liquid cubic" 拒采**：两版论文全文与两官方仓库均无此概念，明确不采纳不臆造；实现选定论文 Lemma 1 Eq.(8) 闭式解 `v_new = A + (v−A)e^{−κ·ts}` + Algorithm 1「LTC 编译为闭式」方案，突触参数化与本库 LTC 完全同构（同 13 参数、同初值区间、erev 固定 ±1 非训练、同 ts 契约、满足 Cell 接口）
- **exprel 稳定化**（任务硬约束）：阈值 1e-2 逐元素分支——小 B 走四阶泰勒（余项 ≤8.3e-13 ≪ float32 ε，梯度在 B→0 存活），大 B 走原式且 B 与分母**解析相消**（图中无除法节点）；阈值处函数值/导数 ~1e-10 连续（有专项跨越扫描测试）；ts 上限 clamp 1e30 保 B 有限非负；ts=1e-40 → v 精确不动（dt→0 正确语义）
- 测试：全 13 参数 gradcheck 最大相对误差 **8.63e-3**（float32 中心差分噪声内）、零掩码纯泄漏闭式回归 1e-4、ts 五类非法值 panic、同种子逐位确定、5 步 BPTT 梯度有限

**P-A（nn 热路径性能，已完成）**：`synapses()` 两级重构——逐突触对（12n−2 节点/次，掩码每突触入图）→ 逐突触前神经元向量化（5n+3 节点/次）+ 构造期稀疏指示矩阵 MatMul 收缩（erev ±1 焙入 num 指示阵）+ 掩码折叠 `wm=softplus(w)⊙mask` 每 Step 一次矩阵 Hadamard（构造期 Const 叶复用，addGrad 形状安全已论证）。
- **基准（−benchtime=100x −count=3 稳定）**：`LTCStep` 7,360→**3,440 allocs/op（−53.3% ✅ 达标）**、ns −23.1%；`UnrollBackward` 120,163→**68,688（−42.8%）**、ns −18.6%。未达 −50% 项已 pprof 剖析：剩余分配 80.4% 为 tensor.New/Clone/broadcast 闭包等**逐节点固定开销**（tensor/autograd 属本轨禁区），−42.8% 已逼近结构上限；进一步需 autograd 层融合反向，建议独立工单（留档#9）
- **正确性四重门禁**：①既有 ltc_test.go 全绿（含零掩码闭式回归）；②自写逐位 oracle `scalarSynapses` 复刻旧循环，严格 `==` 零容差通过（mask∈{0,1}+指示阵升序折叠 ⇒ 结合次序与旧 Add 链完全相同）；③全 15 参数有限差分 gradcheck 最大相对误差 **9.84e-5**；④example 首轮 loss **0.690761 = 0.690761 逐位相等**（优于 1e-4 容差）、终损 0.041996 不变。调试中捕获并修复一个隐蔽 bug：erev 初始化位点前移导致 rng 抽取顺序漂移（首轮 loss 变 0.777），以 HEAD 旧码 dump 15 参数指纹逐位比对定位恢复

**F-A（optimizer 包，已完成）**：6 文件，覆盖率 **100%**，`-race` 干净。`Optimizer` 接口（`Step(params)`，调用方拥有 ZeroGrad 时机——免费支持梯度累积模式）+ 三个显式 struct：`SGD{LR}`、`Momentum{LR,Mu}`（经典重阻尼约定，velocity 存未缩放梯度——与 doc/training.md 范例逐字核对一致）、`Adam{LR,Beta1,Beta2,Eps}`（Kingma & Ba 含偏差校正，`NewAdamDefault` 便捷构造）。超参全导出字段（训练中途热改 LR 有测试背书）；状态按 `*Variable` 指针隔离，同指针改尺寸 panic 而非静默错步；nil-Grad 跳过且不推进 Adam 步数。测试 17 组：SGD/Momentum 手算逐位、Adam 论文 Algorithm 1 独立转写对照（~5 ulp，arm64 FMA 缩合所致，真公式错误远超此量级）、两个收敛测试复现 README quickstart 的 w=2.0000/b=1.0000、`TestMomentumMatchesDocExample` 复现文档算例 w=5.0007、状态隔离/指针键/20 例非法超参 panic。

## 项目终态

| 指标 | 基线（87ccf77） | 终态 |
|---|---|---|
| 编译 | nn ❌ 失败 | ✅ 四包（tensor/autograd/nn/optimizer）全绿 |
| 能力 | 仅 LTC（且前向从未运行） | LTC（向量化热路径）+ **CfC 闭式细胞**（NMI 2022 忠实实现）+ **optimizer 包**（SGD/Momentum/Adam） |
| 测试 | nn 无测试；tensor/autograd 绿 | ✅ 全包 `-race` 绿 |
| 覆盖率 | tensor 85.7% / autograd 97.6% / nn 无 | **tensor 89.5% / autograd 97.8% / nn 98.7% / optimizer 100%** |
| 性能 | 无度量 | LTCStep allocs **−53.3%**（3,440/op）、UnrollBackward **−42.8%**（68,688/op）——红队独立 oracle 同机复测 |
| 红队验证 | 初审 13 项（1C/4H/5M/3L），裁决不可用于生产 | 初审全部销账 + 阶段6三路复审（性能等价/optimizer 更新式/CfC 论文符合度）**全 ✅** |
| 基建 | 非 Git 仓库，无文档 | **已发布 https://github.com/qorm/LNN（public，v0.1.0，`go get` 可用，CI 绿）** · 双语指南 **16 篇**（doc/ 与 doc/zh/ 各八篇）+ 4× godoc · MIT（© 2026 QORM）· Makefile（含 bench）· GitHub Actions CI · 13 项基准 · 2 个 example |
| 端到端 | 无 | LTC example loss 0.691→0.042；CfC 示例 0.621→0.029 |
| 技术债 | 6 项留档 | Div 闭式（−11% 时延）、synapses 向量化、F3 等销账；余项 11 条全部 🟢/Info 级留档 |
| 编排 | — | **19 个 agent / 六阶段**：分析×4（阶段1）· 实施×3（阶段3）· 红队复审×1（阶段4）· 技术债/基建/文档×4（阶段5）· 实施×3 + 红队验证×3 + 文档同步×1（阶段6） |

Git 历史：`87ccf77` 基线 → `08aba45` 阶段3a → `102af40` 阶段3b → `a351daf` 阶段4收尾 → `66d4adf` 阶段5 → `b7f302c` 阶段6a → `fd14fdf` 阶段6c 终局

## 遗留问题登记（技术债留档，均不阻断）

| # | 问题 | 来源 | 严重度 | 处置 |
|---|---|---|---|---|
| 1 | 并发 Backward 数据竞争 | V-04 | 接受风险 | 单线程契约文档化（README + module.go），用户遵守契约即无风险 |
| 2 | ~~`Div` 借 `Pow(-1)`~~ | 核心B/V-11 | — | ✅ **阶段 5 已销账**（闭式单节点，−11% 时延 / −6.7% allocs）；den≈eps 的 1/b² 梯度放大为数学本性，已文档化 |
| 3 | `SumRows→[1,n]` vs `SumCols→[m]`、1D⊕1D→[1,n] 约定不对称 | 核心A/V-12 | Low | API 破坏性，另立评估（README 已披露） |
| 4 | Randn Box-Muller 尾部硬截断 ~7.43σ | V-13/F4 | Low | 对初始化可忽略，采样器用途需留意 |
| 5 | tiny-ts 域（<1e-38）仅保有限性、物理保真退化 | F1 | Info | 正常训练域不受影响，留档 |
| 6 | `Stack` 产出 3D 无消费方、ops.go 五职责 | 核心A/健康度 | 🟢 | Stack 已标注 Experimental；~~无 CI/Benchmark~~ ✅ 阶段 5 已补（GitHub Actions + 13 基准） |
| 7 | nn 热路径分配量（LTCStep 7372 allocs/op、Unroll 12 万 allocs/op） | D2 基准量化 | 🟢 | 已有基准尺子，图算子融合/掩码预施加为后续优化方向 |
| 8 | 三份源码内旧包注释与 doc.go 并存 | D3 发现 | 🟢 | go 1.26 下 doc.go 胜出、用户无感，未来可清理 |
| 9 | Backward 阶段逐节点固定开销（tensor.New 46%/Clone 20%/broadcast 闭包 14%） | P-A pprof | 🟢 | UnrollBackward 再压缩需 autograd 层改动（Sigmoid-Hadamard 融合反向、addGrad 原地写入、去闭包），独立工单候选 |
| 10 | CfC 的 erev/sErev 以 Var 叶入图产生死梯度 | CfC 红队 | Info | LTC 已焙入 Const 指示阵；CfC 可同法优化，正确性无碍（Parameters 排除⇒优化器不可达），微量浪费 |
| 11 | `NewSGD(+Inf)` 被字面校验放行；构造后字段可改非法值 | optimizer 红队 | Info | 信任模型自洽且已文档化，sgd.go 注释已补（6c） |
| 12 | tensor.New 每节点前向输出 + Shape/Data 双分配（终态 pprof 64.9%） | P-B | 🟢 | 进一步压缩需 parents 定长槽化（受阻既有结构断言测试）与 Tensor 定秩 Shape（公共 API 禁区），独立工单候选 |
| 13 | ~~Sigmoid-Hadamard 融合反向~~ | P-B | — | ✅ **阶段 8 已销账**（SigmoidHadamard 算子，−5.6%/−5.8%，结构上限实测终答） |
| 14 | 指示矩阵 O(units³) 实体化（构造器与加载双悬崖） | 总扫红队 F1 | 🟡 | 加载侧已由 maxUnits 上限封堵；根因需稀疏收缩（不实体化 [units²,units]），独立工单候选 |

## 变更日志

- 2026-07-29：建立进度文档；派发阶段 1 的 4 路分析 agent。
- 2026-07-29：阶段 1 完成（4 路回报，红队 13 项实测漏洞）；阶段 2 产出 PLAN.md，建立 Git 基线 87ccf77。
- 2026-07-29：阶段 3a 完成（Agent-T 核心修复 + Agent-I 基建），提交 08aba45。
- 2026-07-29：阶段 3b 完成（Agent-N nn 大修 + examples），主控独立 gauntlet 复核一致，提交 102af40。
- 2026-07-29：阶段 4 红队复审：**V-01～V-13 全部销账，裁决生产就绪（~90%）**；采纳 F2/F5 文档建议并复跑 gauntlet 无漂移；前四阶段关闭。
- 2026-07-29：阶段 5 启动（技术债 + 工程成熟度），应用户要求文档按中英双语交付；D1（Div 闭式/F3/注释化）+ D2（13 基准/CI/make bench）+ D3（英文文档）并行完成；DivDenLoop 复测确认 −11% 时延。
- 2026-07-29：D4 中文镜像文档交付（931 行，全部示例实测），回源核对反查英文版 4 处漂移并由主控双语修正；终检 gauntlet 全绿；前五个阶段关闭。
- 2026-07-30：用户选定双轨并行，阶段 6 启动：6a 三路实施（optimizer 包 100% 覆盖 / CfC 闭式细胞取证实现 / synapses 向量化 −53.3%），提交 b7f302c。
- 2026-07-30：6b 红队三路验证全 ✅——性能组（git 历史 oracle 差分，前向 ≤1.79e-7）/ optimizer 组（教科书参考逐位对照 + 鉴别力校验）/ CfC 组（六路取证 liquid cubic 零命中、方程级忠实裁决、10/10 数值对抗、含自我证伪留档）。
- 2026-07-30：6c 双语文档同步（8 工单 20 文件，反查修正 3 处偏差），主控补修 examples 注释，终检 gauntlet 全绿、旧文案 grep 零残留，提交 fd14fdf；前六个阶段关闭。
- 2026-07-30：阶段 7 双轨再进：7a 序列化（serialize 包 + nn Save/Load）+ autograd 深改（−50.55%）+ CfC 示例（3d47b4c）；7b 红队双路——序列化 7,500 变异体 0 panic 但 F1/F2 资源耗尽（修复：4GiB→33KiB）、autograd 逮住 F1 panic 回归（修复方自证又震出同族 F4-F6，52k 图四类差异归零）（154c9ca）；7c+ 模块迁移 github.com/qorm/LNN；7c 双语持久化指南 + 深改机制文档 + 7 处偏差修正（57e8bc7）。
- 2026-07-30：**发布 https://github.com/qorm/LNN**（合并用户初始提交、LICENSE © 2026 QORM、v0.1.0 标签、CI 3/3 success、Go 代理解析实测通过）；前七个阶段关闭。
- 2026-07-30：阶段 8 v0.2 双轨——8a SigmoidHadamard 融合（−5.6%/−5.8% 逐位）+ 黄金向量/版本策略 + erev 焙入（79d2eea）；8b 红队双路（新代码专项三角证据链全成立 ±0 角除外 / 总扫逮住 F1 units³ 内存悬崖）→ F1 修复 maxUnits/maxInDim=256（0ae7d02）；8c 覆盖率收复 autograd 100%/tensor 99.7%/nn 100% + 变异致死 17/18（e6ccdda）；8d 双语文档五工单 + 4 处反查修正。
- 2026-07-30：**发布 v0.2.0**（tag + 推送；Go 代理解析 v0.2.0 实测通过）；前八个阶段关闭。
- 2026-07-30：**CI 红灯（v0.2.0 首跑即败，本地全绿）**——黄金向量测试在 GitHub amd64 runner 上以 1 ULP 之差失败（0xbe8aa433 vs 0xbe8aa430）。根因：Go 规范允许 FMA 缩合且缩合行为随架构而异（arm64 发射 FMADDS），**浮点逐位等价不是 Go 的跨平台保证**；黄金向量在 arm64 生成却要求 amd64 逐字吻合，断言过严。库代码在规范意义上平台无关、零缺陷；v0.2.0 tag 保留（依赖可用），修复走 v0.2.1：测试按平台分级（arm64 逐位 / 其他平台结构字节精确 + ≤4 ULP 容差 + 鉴别力守门）+ 双语冻结承诺精确化。方法论教训：**本地绿 ≠ 跨平台绿，CI 是验证链不可或缺的一环**——此前所有逐位门禁都在同机成立
