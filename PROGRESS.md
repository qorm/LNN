# LNN 项目实施进度文档

> 主控（orchestrator）维护 · 每阶段结束同步更新
> 项目：`lnn` — 纯 Go 数值计算 / Liquid Neural Network 库（约 1,723 行，3 个包）
> 建立日期：2026-07-29

## 总览

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 并行项目分析（4 路 agent） | 🔄 进行中 |
| 2 | 汇总分析 → 产出 PLAN.md 规划 | ⏳ 待启动 |
| 3 | 按规划实施（目录整顿 + 缺陷修复 + 补测试） | ⏳ 待启动 |
| 4 | 红队复审 + 全量验证 | ⏳ 待启动 |

## 阶段 1：并行分析（进行中）

派发时间：2026-07-29

| Agent | 职责 | 范围 | 状态 |
|---|---|---|---|
| 核心分析 A | 数据结构 / 算子 / 梯度正确性 | tensor/ + autograd/ | ✅ 已回报 |
| 核心分析 B | LTC 语义 / wiring / module 设计 | nn/ | ✅ 已回报 |
| 红队审计 | 对抗性漏洞挖掘（panic / 数值 / 并发 / 供应链），含实际复现 | 全仓库 | 🔄 运行中 |
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

## 遗留问题登记

| # | 问题 | 来源 | 严重度 | 处置 |
|---|---|---|---|---|
| — | （待汇总） | | | |

## 变更日志

- 2026-07-29：建立进度文档；派发阶段 1 的 4 路分析 agent。
