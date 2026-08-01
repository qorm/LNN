# LNN 项目实施规划（PLAN.md）

> 由主控汇总阶段 1 四路分析（核心分析 A/B、红队审计、工程健康度）产出
> 日期：2026-07-29 · Git 基线：`87ccf77`（实施前快照，可随时 `git diff` / 回滚）

## 0. 现状一句话

数值内核（tensor/autograd）正确且测试扎实（gradcheck 全绿、覆盖率 86–98%、零第三方依赖）；
但 **nn 包无法编译、LTC 前向从未被执行过**，存在语义级 bug（erev 误入训练参数）、
多处可由合法输入触发的静默 NaN/梯度腐蚀路径，仓库无版本控制/文档/示例。
红队裁决：**当前不可用于生产**，须修复 V-01～V-05 并补齐 nn 测试后重新评估。

## 1. 目标目录结构

本轮只做**增量式**整顿，不做破坏性重排（三个包平铺于根目录对 Go 库是惯用布局，保留）：

```
LNN/
├── go.mod
├── README.md                 ← 新增：定位、安装、最小示例、适用规模与并发契约
├── LICENSE                   ← 新增：MIT（版权占位，可替换）
├── .gitignore                ← 新增（已随基线提交）
├── Makefile                  ← 新增：fmt / vet / test / cover 一键目标
├── PLAN.md  PROGRESS.md
├── tensor/
│   ├── tensor.go  ops.go  random.go
│   └── tensor_test.go        ← 扩充：构造校验、panic 契约、红队 PoC 回归
├── autograd/
│   ├── variable.go  ops.go
│   └── ops_test.go           ← 扩充：idx 别名、二次 Backward、Pow(0) 回归
├── nn/
│   ├── module.go             ← 修订：删除 CfC/RNN 虚假承诺，补并发契约说明
│   ├── cell.go               ← 新增：Cell 接口 + 通用 Unroll
│   ├── linear.go  ltc.go  wiring.go
│   ├── ltc_test.go           ← 新增
│   └── wiring_test.go        ← 新增
└── examples/
    └── ltc-sequence/main.go  ← 新增：端到端玩具训练回路（手搓 SGD），兼作集成冒烟
```

**明确不做（本轮）**：`ops.go` 拆分四文件、`SumRows/SumCols` 与 1D→`[1,n]` 广播约定统一
（均为 API 破坏性变更，nn 现有代码依赖现行约定，另立评估）；CI、Benchmark、CfC 实现。

## 2. 工作项（按优先级）

### P0 — 阻断性与语义正确性
| # | 工作项 | 依据 | 验收 |
|---|---|---|---|
| P0-1 | 修 `nn/ltc.go:131` 编译错误：`synapses()` 统一为「接收完整矩阵、内部按行 SliceRow」，删除 recurrent 侧预切数组与函数内死分支 | V-01（Critical） | `go build ./...` 绿；LTC 前向首次真正可运行 |
| P0-2 | `erev/sErev` 从 `Parameters()` 剔除，降级为非训练常量（论文语义 ±1） | 核心B-🔴2 | `Parameters()` 不含 erev；测试断言之 |
| P0-3 | `addGrad` 增加 `SameShape` 断言（元素数同而形状异 → panic 带上下文） | 核心A-🔴、V-12 | 回归测试：`[1,6]`⊕`[2,3]` panic |
| P0-4 | `GatherRows` 入口拷贝 `idx`，闭包捕获副本 | V-08（实测梯度腐蚀） | 回归测试：前向后改 idx，梯度不变 |
| P0-5 | `Backward` 结束后将非叶节点 `Grad` 置 nil（或显式禁止二次调用） | V-09（实测 3 倍超线性累加） | 回归测试：同图二次 Backward 行为确定 |

### P1 — 数值安全护栏
| # | 工作项 | 依据 |
|---|---|---|
| P1-1 | `LTC.Step` ts 校验改 `!(ts > 0)`（NaN 感知）；`cm·unfolds/ts` 以 float64 计算并 clamp 到 float32 有限域，保证有限输入→有限输出 | V-02/V-03（ts=1e-40 与 NaN 实测全 NaN） |
| P1-2 | tensor 构造校验：维度 `d ≥ 0`；`Size()` 用溢出安全乘法（`math/bits`），越界 panic 带语义信息 | V-05（2⁶² 幽灵张量）/V-06（`New(-2,-3)`） |
| P1-3 | `SoftmaxRows/LogSoftmaxRows` 空行（n=0）提前返回空结果；`MeanAll` 空张量显式 panic 而非 NaN | V-07/V-13 |
| P1-4 | `Pow` 反向 `p==0` 特判导数为 0（消除 0×Inf=NaN） | V-11、核心A |
| P1-5 | `NewLTC` 形状校验只比 `Shape` 字段，不分配张量 | V-10（校验即 OOM） |
| P1-6 | `wiring`：概率值域 `[0,1]` 校验、维度非空校验；掩码字段收敛为方法访问（防外部篡改） | 核心B-🟡8/9/11 |

### P2 — 测试、文档与端到端
| # | 工作项 |
|---|---|
| P2-1 | 新增 `nn/ltc_test.go`：前向冒烟、全零掩码退化为纯泄漏的闭式回归、跨 unfolds 梯度有限性、ts 校验、erev 不可训练、种子确定性 |
| P2-2 | 新增 `nn/wiring_test.go`：p=0/p=1 边界、非法概率 panic、维度校验 |
| P2-3 | 补齐 tensor/autograd 测试盲区：panic 契约（MatMul/SliceCol/broadcast/FromRows/At-Set）、Randn 奇数分支、手动 seed Grad、`[m,1]` 广播 Sub 反向；红队 PoC 全部转为回归测试 |
| P2-4 | 新增 `nn/cell.go`：`Cell` 接口 + `Unroll`；修订 `module.go` 文档（删虚假承诺、写明单线程契约 V-04） |
| P2-5 | 仓库基建：README（含可运行最小示例与适用规模说明）、LICENSE(MIT)、Makefile |
| P2-6 | `examples/ltc-sequence`：端到端训练回路，兼作集成冒烟 |
| P2-7 | 全部触及文件 `gofmt -w` |

### 红队误报/降级项（不修，留档）
- 「递归 Backward 栈风险」（核心A）→ 红队 10 万层实测安全（Go 动态栈），降级为**非问题**
- `Div` 借 `Pow(-1)`、`inW/outW` 初值不对称、`Stack` 孤立 API、ops.go 拆分 → 记录为技术债，不阻断本轮

## 3. 执行编排（阶段 3）

```
3a（并行）┬─ 实施 Agent-T：tensor + autograd 修复（P0-3/4/5、P1-2/3/4、P2-3 部分）+ 自触文件 gofmt
         └─ 实施 Agent-I：基建（README/LICENSE/Makefile）—— 不碰任何 .go 文件
3b（3a 全部完成后）─ 实施 Agent-N：nn 修复（P0-1/2、P1-1/5/6、P2-1/2/4/6）+ 依赖上游 API 稳定
阶段 4 ─ 红队复审 Agent：对 V-01～V-13 逐项核验 + 新一轮对抗；主控运行全量 gauntlet
```

## 4. 全局验收标准（DoD）

- `go build ./... && go vet ./...` 全绿；`gofmt -l .` 输出为空
- `go test ./... -count=1 -race` 全绿；nn 包覆盖率 ≥ 70%
- 红队 V-01～V-10 每一项都有对应回归测试
- `go run ./examples/ltc-sequence` 可运行且 loss 下降
- PROGRESS.md 全程同步，遗留问题登记表清零或明确留档

---

# 阶段 16 规划（2026-07-31 追加）

> 依据：阶段 15 双线调研综合机会地图（详见 PROGRESS.md 阶段 15 结论）；库主拍板**三线全做**。
> 基线：HEAD `5d9d7ce`（v0.4.3 后文档态），`go build/vet/gofmt` 干净，五包 `-race` 全绿。

## 三线工作项

| 线 | 内容 | 依据（阶段 15） | 验收 |
|---|---|---|---|
| ① remat + Detach | `autograd.Detach` 原语 + 细胞级分块 BPTT（段逆序重算，复用 Step+Backward） | 内外重合点 ①：1000 步 420MB 悬崖 → O(chunk)；Revolve/Chen 学术根基；重算确定性 ⇒ 逐位近免费 | 参数与 h0 梯度对全展开路径**逐位相等**（差分测试多配置）；峰值内存 O(chunk) 实证；长序列能力解锁 |
| ② LTC 融合自定义算子 | 融合前向 + 手写闭式 VJP，消除每节点解释开销（墙钟 ~80% 所在） | 内外重合点 ②：原型前向 197µs→74.5µs（2.6×）；唯一真墙钟杠杆；API 零变更 | **逐位等价优先**：只消解释开销、不改运算次序；前向+梯度对全展开/remat 双 oracle 逐位差分；五基准提速复测 |
| ③ PGO + SSM 定位 | PGO 实测收益量化 + 库形态下可交付形态（文档工作流/make 目标）；README/文档接入「非线性 SSM」定位叙事（零代码） | 外独有：PGO 2–14% 近零成本；战略发现：液态网络被重铸为非线性 SSM，本库 CfC 与 Mamba 步结构同构 | 收益数字实测背书（不虚报）；事实断言逐条一手出处；双语同构、零断链 |

## 执行编排

```
16a（并行）┬─ Agent-R：Detach + 分块 BPTT（autograd + nn + 测试/基准）
          └─ Agent-D：PGO 实测与交付形态 + SSM 定位双语文档（不碰库 .go）
16b（16a 完成后）─ Agent-F：LTC 融合算子（以全展开 + remat 双路径为逐位 oracle）
16c ─ 红队多路复审（逐线对抗 + 总扫）+ 主控全量 gauntlet
16d ─ 双语文档同步 + PROGRESS 定稿 → 呈库主请示发布
```

## 阶段验收（DoD）

- 既有门禁全绿：build/vet/gofmt 空/五包 `-race`/fuzz-smoke 8 目标
- 新代码覆盖率向库标准看齐（正当 100% 或诚实留档）
- 线 ①② 逐位等价差分测试常驻化（oracle = 全展开路径）
- 基准数字主控独立复验；文档断言回源核对
- PROGRESS.md 逐阶段同步；git 提交/打 tag 前逐项请示库主
