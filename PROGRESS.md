# LNN 项目实施进度文档

> 主控（orchestrator）维护 · 每阶段结束同步更新
> 项目：`lnn` — 纯 Go 数值计算 / Liquid Neural Network 库（约 1,723 行，3 个包）
> 建立日期：2026-07-29

## 总览

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 并行项目分析（4 路 agent） | ✅ 完成 |
| 2 | 汇总分析 → 产出 PLAN.md 规划 | ✅ 完成（见 PLAN.md） |
| 3 | 按规划实施（目录整顿 + 缺陷修复 + 补测试） | 🔄 进行中 |
| 4 | 红队复审 + 全量验证 | ⏳ 待启动 |

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
| 实施 Agent-N | nn 修复（P0-1/2、P1-1/5/6、P2-1/2/4/6）——等 3a 完成后派发 | ⏳ 待启动 |

### 实施回报摘要

**Agent-I（基建，已完成）**：新增 `README.md`（206 行，英文）/ `LICENSE`（MIT）/ `Makefile`（fmt/vet/test/cover/build/all，tab 缩进已自检）。README 最小示例在 /tmp 独立模块**实测跑通**（loss 1.399→0.000，恢复 w≈2.0000、b≈1.0000）；如实披露 nn 编译失败/CfC 未实现/单线程契约（援引 V-04）。附报告 6 处 API 与文档不一致（均已在既有工作项覆盖，无新增工作项）。

**Agent-T（tensor+autograd，已完成）**：11 项工作全部落地，每项附回归测试：
- P0-3 `addGrad` 形状断言（panic 信息含双形状）；P0-4 `GatherRows` idx 入口拷贝（红队错位梯度 [0 1 1 0] → 正确 [1 0 0 1] 回归）；P0-5 `Backward` 后非叶节点 Grad 置 nil（同图二次反向从红队 3 倍超线性 → 精确 2 倍线性叠加）
- P1-2 `Size()` 用 `bits.Mul64` 溢出检查 + `New` 负维度校验（幽灵张量 PoC 转回归）；P1-3 0 列 Softmax 返回空结果、`MeanAll` 空张量显式 panic；P1-4 `Pow` p==0 导数特判为 0
- P2-3 盲区清零：panic 契约表驱动 8 子测试、Randn 奇数分支+方差、手动 seed Grad 路径、Sub 列广播 gradcheck；`Uniform(lo>hi)` 保留镜像行为并文档化（向后兼容决策）
- 验证：`gofmt -l` 空、`go vet` 零告警、`go test -race` 全绿；覆盖率 **tensor 85.7%→89.5%**、**autograd 97.6%→97.7%**；范围自律（未触碰 nn/ 与文档）

## 遗留问题登记

| # | 问题 | 来源 | 严重度 | 处置 |
|---|---|---|---|---|
| — | （待汇总） | | | |

## 变更日志

- 2026-07-29：建立进度文档；派发阶段 1 的 4 路分析 agent。
