# FMA V5 对上传方案的需求追踪矩阵

评估日期：2026-07-24  
对照基线：上传的《前沿数学建模 Agent 搭建方案》  
评估对象：`D:\modeling` 当前 V5 Graph-native 实现与 Iteration 28 冻结证据

## 1. 结论与口径

当前 V5 已经形成一个较强的可信控制内核，但尚不是能够独立解决前沿真实问题的数学建模 Agent。

- 控制面已经覆盖：内容寻址、精确文件快照、事件链、认证门禁、撤销闭包、独立评审收据、科学 adapter 收据、raw-data 基线、预测冻结、论文构建和 single-writer 并发保护。
- 科学能力仍主要停留在协议与 adapter 边界：没有生产级守恒、收敛、Markov、Sobol、ensemble、extrapolation、UDE 或 SINDy 实现。
- 自主能力尚未验收：没有由外部 Codex/模型 driver 在未见真实任务上完成 S0--S6 的能力证据。
- 测量仪器只完成了部分基础设施：没有 gold 阶段注入、真实机制开关运行、归因矩阵或机制增益实验。

本文使用三种状态：

- `implemented`：该要求的核心机制已经由当前代码实现，并有与其声明范围相称的测试或运行证据。
- `partial`：已经有结构、接口或部分控制机制，但上传方案的完整活动或验收条件尚未满足。
- `deferred`：尚无实质实现，或只有未来设计说明而没有当前可执行证据。

`implemented` 只表示相应控制机制已经实现，不表示科学正确性、现实外部有效性或真实 Agent 能力已经得到证明。

## 2. A1--A5、W1--W4、R1

| 要求 | 当前状态 | 已实现证据 | 未满足内容 |
|---|---|---|---|
| A1 制造验证信号 | `partial` | `CheckRegistryV50`、`AdapterExecutionReceiptV50` 和 `CheckResultV50` 强制区分 scientific computation、integrity、presence 与 judgement；L0--L4 缺 adapter 时记录 `NOT_RUN` 并阻断 gate | 没有生产级守恒、收敛、MMS、Markov、Sobol 等 adapter；没有实现方案要求的全部命名检查 |
| A2 误差杠杆前置 | `partial` | S1 要求三个结构不同的候选；候选结构哈希排除 ID、标签与 lineage，防止改名伪造竞争；候选与 modeler execution receipts、最终 `ModelSpecV50` 精确绑定 | 没有真实运行三个独立模型上下文；没有强制 A/B/C 约束变体、盲 referee 或多轮候选演化 |
| A3 文件即记忆 | `implemented` | 工作文件被快照为内容寻址 graph artifacts；stage manifest 绑定路径、字节、前序 gate；事件链、stale 检测、撤销谱系和 single-writer 锁均已实现 | 该机制只保证状态、历史和来源完整性，不保证文件中的科学内容正确 |
| A4 独立上下文审稿 | `partial` | review receipt 绑定 producer/reviewer run、不同 context、精确输入哈希、transport trace、输出工件、verdict 和 HMAC | context ID 与隔离声明本身不能证明真实 fresh-process 或独立模型执行；synthetic fixture 中的上下文由同一 runner 构造 |
| A5 阶段内自主、阶段间门禁 | `partial` | S0--S6 均是 `work node -> evaluation node`；下一阶段依赖当前 gate；HMAC certificate、撤销闭包、重试 lineage 与 single-writer transition 已实现 | 跨阶段门禁已实现，但阶段内自主 Agent driver、工具调度和长任务恢复尚未在 V5 接入 |
| W1 状态抽象可检验 | `partial` | ValidationPlan 和 adapter registry 可以登记 L3 Markov obligation；Iteration 28 有 synthetic Markov fixture | 没有生产级 GBDT A/B、blocked bootstrap 或真实状态遗漏阳性对照 |
| W2 价值等价 | `partial` | S0 有安全算术 decision/loss DSL 与 canary；S5 的每条决策断言必须追溯到 ResultIndex 与 UQ claim | 真实任务尚未按下游决策质量评价模型；external scorer 没有执行 S0 的 decision function |
| W3 防模型剥削 | `partial` | ValidationPlan 冻结 ensemble disagreement threshold 和 out-of-support action；S5 会重算 disagreement，越界或高分歧时强制 `return_to_data_acquisition` | 没有真实 ensemble、信任域、目标区域采数、重新训练和重新验证的 Dyna 闭环 |
| W4 灰盒谱系导航 | `partial` | scaffold 包含 `regime-diagnosis` 与 `methods-mechanistic` P0 skills，能够记录机理、学习闭包与数据驱动路线假设 | 没有 `methods-greybox`、`methods-uq`、UDE、SINDy 或灰盒真实系统适配器 |
| R1 一切可归因 | `partial` | `WorkflowProfileV50` 定义 GATE、REDTEAM、COMPETE、CHECKS、BUILDPAPER、SKILLS、ENSEMBLE 身份；external harness 可检查 arm 声明和明显 no-op | 当前 runtime 只允许完整 profile；off profile 被拒绝；`assess_ablation` 固定为不认证；没有 gold 阶段注入、配置矩阵或真实归因实验 |

## 3. S0--S6 工作流

| 阶段 | 当前状态 | 已实现证据 | 未满足内容 |
|---|---|---|---|
| S0 体制诊断 | `partial` | `RegimeDiagnosisV50`、`DecisionFunctionSpecV50`、无副作用算术解释器和 canary；P0 skill 要求数据丰富度、平稳性、查询类型与下游决策 | 数据丰富度和平稳性仍主要依赖文字；没有真实 literature-scout 检索、来源抓取与检索质量验证 |
| S1 多候选形式化 | `partial` | 三候选结构去重；modeler/scout receipts；assumption failure/falsification/abandonment；symbol units/bounds；两个以上 limit cases；identifiability risks；referee 与 red-team gate | 没有 Buckingham Pi 或无量纲化字段；没有真实独立 modeler、约束变体、盲 referee 和实际文献依据 |
| S2 数据获取与审计 | `partial` | harness 在 S2 前冻结 raw baseline；raw、transform script、transform params、processed artifact 形成路径和哈希闭包；ledger IDs 与 processed IDs 精确相等；synthetic/estimated 项必须声明不可得原因并绑定 sensitivity obligation；data-auditor gate | 不验证 URL 可达性、许可真实性或数据实际获取过程；不能证明模型和论文中的每个数值字面量都有 ledger 行 |
| S3 实现与求解 | `partial` | `CodeManifestV50` 绑定 source tree、environment、replay receipt、Fermi estimate、toy oracle；result index 与实际 JSON payload 精确绑定；L0--L2 必须经 adapter receipt | replay receipt 是冻结工件而不是通用 clean-environment runner；没有生产级 convergence、MMS、粗精度到全精度计算阶梯和后台恢复 |
| S4 验证与 UQ | `partial` | frozen ValidationPlan、L1--L4 adapter receipts、UQ interval 存在性和单位检查、support status、disagreement threshold、red-team gate | plan 当前只强制相应 level 至少有一项适用检查，不强制方案列出的全部命名 L1--L4 检查；没有生产科学 adapter |
| S5 使用与外循环 | `partial` | assertion 到 result/UQ 的机械追踪；high-disagreement 由 frozen threshold 重算；out-of-support 强制返数据阶段；有 S4-bound `PredictionSealV50` API | `return_to_data_acquisition` 仍是决策状态，不会自动完成采数和重跑；V5 prediction seal 尚未与 external harness registration 做跨链认证 |
| S6 构建式成文 | `partial` | 论文模板必须使用 result placeholder；硬编码数字被拒；`results/values.json` 必须是 ResultIndex 的精确投影；真实 `pdflatex` 构建；receipt 绑定模板、结果、TeX 和 PDF；final red-team gate | 图在没有 generator-bound figure manifest 时被 fail-closed 禁止，尚未实现图生成链；未检查假设编号引用；未强制 writer role execution receipt；不验证所有语义结论是否受证据支持 |

## 4. L0--L5 验证金字塔

| 层级 | 当前状态 | 已实现证据 | 未满足内容 |
|---|---|---|---|
| L0 可执行性 | `partial` | code/environment/replay receipt 的路径和哈希闭包；L0 PASS 必须在 adapter evidence 中引用冻结 replay receipt hash | 没有通用干净环境一键重跑器；没有对全部 `src/` 输出进行重算和结果哈希比较 |
| L1 结构不变量 | `partial` | symbols 可以声明单位和 bounds；model spec 可以声明 conservation laws 与至少两个 limit cases；有 task-specific adapter 接口 | 没有生产级 dimensional、conservation、bounds、limits 检查器 |
| L2 已知解与数值正确性 | `partial` | toy oracle 路径和哈希被 CodeManifest 冻结；L2 adapter 缺失会阻断 S3 | 没有通用 toy 对拍框架、收敛阶检查或 PDE MMS 实现 |
| L3 模型充分性 | `partial` | ValidationPlan 和 adapter registry 支持 L3 obligation；Iteration 28 有 synthetic fixture evidence | 没有方案要求的 Markov GBDT/bootstrap、residual structure、baseline duel、cross-model 生产实现 |
| L4 不确定性与稳健性 | `partial` | UQ schema、finite interval、support status、ensemble disagreement、frozen threshold 与 synthetic sensitivity obligation 已实现 | 没有 Sobol/Morris、至少五成员 ensemble、凸包/密度支撑域计算、完整 quantitative-claim coverage |
| L5 诚实性与一致性 | `partial` | raw-data baseline、ledger/path/hash closure、硬数字拒绝、result projection、paper build receipt 与 prediction integrity 已实现 | 不检查 URL 可达性；未实现 figure lineage 和 assumption reference audit；不能自动证明每个入模数字均有 ledger 行或论文语义无越界 |

## 5. M0--M4 里程碑验收

| 里程碑 | 当前状态 | 已完成 | 未满足验收 |
|---|---|---|---|
| M0 最小闭环 | `partial` | workspace scaffold、AGENTS constitution、Makefile facade、三个 P0 skills、S0--S6 synthetic fixture 和真实 PDF 构建 | `methods-mechanistic` 明确不是 98 个 HMML 模板；没有 MM-Bench 题；没有 LLM/Codex 自主无人工运行；没有生产 L0--L2 adapters |
| M1 机制硬化 | `partial` | Graph/HMAC 门禁强于原始 hook/stamp；referee/red-team/data-auditor receipts；raw stamp、伪 gate、伪 scientific PASS、raw mutation、硬编码数字等测试 | 没有真实 subagent transport；URL audit 和 figure/assumption L5 未完成；没有裸 Agent 对照或同题可测增益；authority key 仍需外部进程隔离 |
| M2 世界模型层 | `partial` | L3/L4 plan、adapter boundary、UQ/support/disagreement 协议存在 | 没有生产 L3/L4 checks、greybox/UQ skills、SIR/电池真实纵切片、UDE-to-SINDy、状态遗漏阳性对照 |
| M3 测量仪器 | `partial` | external public/private projection、不可变预测快照、holdout 后揭示、机械评分、篡改归零、事件链和 single-writer 已实现 | 没有批量 Agent runner、三级任务集、gold 阶段注入、双层 judge、过程 transcript/cost、归因矩阵或真实机制消融 |
| M4 高难靶点 | `deferred` | 无当前验收证据 | 尚未接入活预测未来揭晓、论文复现集或显著机制变体研究 |

## 6. Iteration 28 的证据边界

Iteration 28 的 final post-amendment 冻结运行证据位于：

- `experiments/iteration_28/PREREGISTRATION.md`
- `experiments/iteration_28/PROTOCOL.json`
- `experiments/iteration_28/run_iteration_28.py`
- 最终本地运行定位符与汇总：
  `tmp/iteration28_final_e2e_7/SUMMARY.json`

`e2e_1`--`e2e_6` 均为已披露的 development smokes，用于在开发过程中
暴露并修复接口、证据闭包和运行时来源绑定问题；它们不是最终发布证据，
也不应与 `e2e_7` 合并计为多次独立成功。

当前结果：

- S0--S6 七个 gate 全部打开；
- 总耗时 48.098251 秒；
- graph verification 与 reopen replay 均通过；
- S2 raw input baseline 被认证；
- S3 L0 evidence 绑定 replay receipt hash；
- 真实 `pdflatex` PDF 构建一致；
- external event chain、预测冻结和注册后篡改检测通过；
- promotion count 为 0；
- runtime source manifest 包含 103 个受监控源码条目；
- 运行开始与结束时各记录 84 个已加载的仓库模块，且
  `loaded_modules_covered_by_manifest=true`；
- 开始与结束的受监控源码树哈希均为
  `7923d648a263a0849f668dbe187e9d392d044a9376826f61426557d8fe45284f`；
- `unchanged_during_run=true`。

该 runtime source snapshot 证明受监控的磁盘仓库源码在本次运行期间未变，
并覆盖当时记录到的已加载仓库模块。它不证明已加载 bytecode、第三方依赖、
进程隔离或主机级 attestation。

但它的证据范围严格是：

```text
fixture_class = synthetic_authenticated_fixture
claim_scope = workflow_control_fixture_only
scientific_qualification_granted = false
real_world_action_authorized = false
capability_claim_permitted = false
host_secrecy_attested = false
cross_chain_binding_verified = false
```

因此 Iteration 28 只证明当前冻结 synthetic control fixture 的控制链能够运行。它不是 MM-Bench 结果，不是未知真实任务结果，不证明 production scientific adapters 正确，也不证明 Agent 已具备自主解决前沿数学建模问题的能力。

## 7. 当前验证证据

当前聚焦回归：

- V5 stage workspace：17/17 PASS；
- V5 external harness：6/6 PASS；
- V5 paper：11/11 PASS；
- V5 scaffold：14/14 PASS；
- single-writer concurrency：5/5 PASS。

这些测试证明相应控制协议和回归行为，不替代完整仓库兼容性回归、真实数据验证、私有 scientific promotion 或人类最终决策。

## 8. 不能据当前证据声称的内容

1. 不能声称 V5 已经能够独立解决前沿或任意真实数学建模问题。
2. 不能把 adapter receipt 等同于科学正确性；它证明登记的 adapter 被调用并提交了证据，结论强度仍取决于 task-specific adapter。
3. 不能把 context ID 或 review receipt 单独当作真实独立进程证明。
4. 不能把 single-writer 锁当作权限隔离；它防止并发写分叉，但不阻止能读取 authority key 的进程调用权威 API。
5. 不能声称 V5 与 external prediction harness 已跨链认证；Iteration 28 明确记录 `cross_chain_binding_verified=false`。
6. 不能声称机制消融已实现；当前 runtime 不运行 off profiles，external ablation assessment 固定不签发有效因果消融。
7. 不能把 synthetic fixture 的 L0--L4 PASS 外推为真实系统科学能力。

## 9. 下一验收顺序

1. 关闭 M0：接入外部 Codex/模型 driver，在一个全新未见公开任务上运行真实独立角色进程，并安装该任务的生产 L0--L2 adapters。
2. 加固 M1：把 authority key、private holdout、reviewer 和 adapter 执行放入低权限独立进程或容器；补 URL、figure lineage 和 assumption reference L5 检查。
3. 关闭 M2：选一个公开 SIR 或电池退化系统，实现真实 Markov、baseline、sensitivity、ensemble、support-domain 与 UQ 检查，并做状态遗漏阳性对照。
4. 关闭 M3：实现真正执行不同 profile 的 runner、gold 阶段注入、归因/消融矩阵、transcript/cost 归档和跨链 prediction seal。

当前最准确的成果定位是：

> FMA V5 已经完成一个保留上传方案设计精髓、且比文本门禁和裸 stamp 更可信的 Graph-native 数学建模控制内核；真实科学能力、端到端自治能力和机制归因能力仍需通过外部 Agent driver、生产 adapter 和未见真实任务补齐。
