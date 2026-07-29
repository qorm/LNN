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

## 项目终态

| 指标 | 基线（87ccf77） | 终态 |
|---|---|---|
| 编译 | nn ❌ 失败 | ✅ 全包绿 |
| 测试 | nn 无测试；tensor/autograd 绿 | ✅ 全包 `-race` 绿（44+ 测试） |
| 覆盖率 | tensor 85.7% / autograd 97.6% / nn 无 | **tensor 89.5% / autograd 97.7% / nn 98.3%** |
| 红队漏洞 | 13 项（1C/4H/5M/3L），裁决不可用于生产 | 全部销账，裁决生产就绪（~90%） |
| 基建 | 非 Git 仓库，无文档 | Git 4 提交 · README（示例实测）· MIT LICENSE · Makefile · examples |
| 端到端 | 无 | example 实测 loss 0.691→0.042（降 94%） |

Git 历史：`87ccf77` 基线 → `08aba45` 阶段3a → `102af40` 阶段3b → 终局文档收尾提交

## 遗留问题登记（技术债留档，均不阻断）

| # | 问题 | 来源 | 严重度 | 处置 |
|---|---|---|---|---|
| 1 | 并发 Backward 数据竞争 | V-04 | 接受风险 | 单线程契约文档化（README + module.go），用户遵守契约即无风险 |
| 2 | `Div` 借 `Pow(-1)`，den≈eps 时梯度 ~1e16 | 核心B/V-11 | Low | 技术债，建议未来闭式实现 |
| 3 | `SumRows→[1,n]` vs `SumCols→[m]`、1D⊕1D→[1,n] 约定不对称 | 核心A/V-12 | Low | API 破坏性，另立评估（README 已披露） |
| 4 | Randn Box-Muller 尾部硬截断 ~7.43σ | V-13/F4 | Low | 对初始化可忽略，采样器用途需留意 |
| 5 | tiny-ts 域（<1e-38）仅保有限性、物理保真退化 | F1 | Info | 正常训练域不受影响，留档 |
| 6 | `Stack` 产出 3D 无消费方、ops.go 五职责、无 CI/Benchmark | 核心A/健康度 | 🟢 | 后续迭代候选 |

## 变更日志

- 2026-07-29：建立进度文档；派发阶段 1 的 4 路分析 agent。
- 2026-07-29：阶段 1 完成（4 路回报，红队 13 项实测漏洞）；阶段 2 产出 PLAN.md，建立 Git 基线 87ccf77。
- 2026-07-29：阶段 3a 完成（Agent-T 核心修复 + Agent-I 基建），提交 08aba45。
- 2026-07-29：阶段 3b 完成（Agent-N nn 大修 + examples），主控独立 gauntlet 复核一致，提交 102af40。
- 2026-07-29：阶段 4 红队复审：**V-01～V-13 全部销账，裁决生产就绪（~90%）**；采纳 F2/F5 文档建议并复跑 gauntlet 无漂移；全部阶段关闭。
