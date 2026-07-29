# FMA 迭代待办

## 大迭代 1

- [x] 结构化 CSV 历史序列契约与 fail-closed 数据质量门。
- [x] 至少三个机制不同的确定性候选，另设 last-value 基线。
- [x] expanding-window rolling-origin evaluator，确保任何预测只使用目标时点之前的数据。
- [x] 经验残差区间及持出覆盖率；明确不宣称无条件 conformal 保证。
- [x] 容量决策 regret、候选间行动一致性与 `NEEDS_EVIDENCE`。
- [x] 稳定、机制错配、结构突变、泄漏/篡改负例。
- [x] 全量回归和本地学习文档更新。

## 大迭代 2 候选

- [x] 选择至少两个有明确来源和语义的官方历史域，冻结原始响应与 cutoff。
- [x] 加入季节尺度/局部窗口演化和冻结漂移诊断。
- [x] 建立跨 series/site 基线、三 seed moving-block bootstrap 与效应区间。
- [x] 把证据撤销传播到 validation/decision 证书。
- [ ] 只有单循环评测出现可复现瓶颈后，才评估并行 worker 或多 agent。

## 大迭代 3 候选

- [x] `EpistemicGraph` 最小可执行子集：主张、证据、反驳、supersedes、effective status。
- [x] 数据修订/撤销后，自动使 validation、skill、shift 和 decision 报告失效。
- [ ] 面向水文失败补充外生变量/状态空间骨架前，先冻结新的问题与数据契约。
- [x] 用隐藏 WorldPack 测“骨架检索 + 演化算子”是否优于直接候选生成，并做成本归一化消融。

## 大迭代 4 候选

- [x] 不追溯改写 V2.2 的 `candidate_rejected`；冻结新的 V2.3 前瞻确认协议。
- [x] 用宏平均配对增益置信下界、逐机制非劣界和负迁移率一侧上置信界替代未经效用解释的原始 strict-win 比例。
- [x] 使用全新未见 seed 扩到足以约束低负迁移率的 case 数；policy、预算、选择器和 private outer 全部预先冻结。
- [x] V2.3 因 1/80 负迁移导致上界 5.79% 而失败，已将该失败签名转为 policy 结构演化。

## 大迭代 5

- [x] 在相同 4 候选预算下，把 global trend 放回 memory policy，并将 SES 从该 exact policy 退役而不删除知识工件。
- [x] 用第二组 80 个全新 private-outer case 重跑未改变的 V2.3 门。
- [x] exact safe policy 通过宏平均、逐机制非劣、负迁移率上界和同预算门；只生成 `synthetic_forecast_worldpack_v23` qualification。

## 下一跨域迭代

- [x] 停止继续扩展同一预测 WorldPack；冻结第二个结构不同的建模家族（动力系统机制发现）。
- [x] 为新家族建立 typed IR、结构候选、经验可识别性、仿真/反事实检查和 private-world generator。
- [x] 测试“骨架检索 + 受控演化”是否跨 IR 方言产生增益；结构恢复有增益，但轨迹/反事实安全门失败，未宣称通用建模 Agent。

## 大迭代 7 候选

- [x] 实现诚实命名的 window-integral matching，与 Savitzky–Golay 点导数完成单组件 private-outer 消融；确认失败，未冒充完整 Weak SINDy。
- [x] 增加 dependence-aware moving-block 参数稳定性门；确认 integral 23/80 case 不稳定，未生成 qualification。
- [ ] 将长程稳定性、动作成本与反事实风险前移到 acquisition，而不只在 hidden probe 事后发现。
- [x] 加入首个受控初值 reset 与主动实验 WorldPack；同预算 80-case 确认通过，仅产生 synthetic initial-condition-design qualification。
- [ ] 接入符号 observability/structural-identifiability 工具前，保持所有参数唯一性主张为 `NEEDS_EVIDENCE`。

## 大迭代 9 候选

- [x] 从固定 ProblemContract 流水线升级为 `MissionConstitution + immutable/supersedable EpisodeProblemContract`。
- [x] 实现类型化认识动作、代码拥有的权限/成本决策、Reality Interface result 和父契约/触发证据 lineage。
- [x] 用同成本 private WorldPack 比较“继续采数据”与“澄清问题语义”；V3.0 一步失败后只演化 action horizon。
- [x] 80-case 全新确认通过宏效应、逐机制非劣、负迁移、零误重构和现实动作禁用门；qualification 仅限 synthetic V3.0.1 scope。

## 大迭代 10 候选

- [x] 将 V3 认识动作 IR 接入已知 actuator map 的有界分段常值输入；与初值 reset 分开消融，不推断未知执行器。
- [x] 把决策敏感的模型判别、D-optimal/信息增益、动作成本和经验风险统一到 acquisition receipt。
- [x] 加入无 admissible 动作时的代码级弃权；明确 bootstrap/ensemble envelope 不是安全保证。
- [x] 构造含错误问题定义、数据校准失败、二次模型库错配和主动干预的跨层 private WorldPack。
- [x] V3.1 两步失败后只演化 horizon 为 V3.1.1；均值修复但负迁移/模型路由仍失败，未生成 qualification。

## 大迭代 11 候选

- [ ] 只演化 acquisition：用目标 probe 的预期后验风险或 goal-oriented A-opt 替代启发式加权；不得同时改 router。
- [ ] 在全新探索 seed 上校准后冻结一次性 confirmation；继续保留随机同预算基线和物理动作等价。
- [ ] acquisition 收口后再单独演化模型错配 router：加入外推持出或跨激励残差，不用训练残差自证。
- [ ] 把 action debt / 最低实验充分性变成通用预算规划器，而不是每个方言手写 horizon。
- [ ] 接入现实数据前先定义 actuator 来源、可撤销证据、操作审批和形式化/经验安全等级。

## 大迭代 14

- [x] 冻结显式 applicability state 与三族 model challenge 协议。
- [x] 用 shared observation/private probe 检验固定二次骨架与 challenge selector；正式失败且保留完整工件。
- [x] 将 challenge failure 转成 typed disposition，而不是继续调 scalar cutoff。
- [x] 捕获 generator/evaluator 共享 target-authority 盲点，新增独立 contract 重建回归。
- [x] 在 fresh 64 cases 上通过五类 disposition 路由；仅授权下一次合成 action experiment。

## 大迭代 15 候选

- [x] 执行 `clarify_decision_target`：39/39 合成动作完成，target accuracy 从 1/3 到 1，contract lineage 与 target-conditioned challenge 可独立重算。
- [x] 执行 `acquire_target_discriminating_evidence`：fresh 22-case 同预算双臂均 0 resolved；候选被 refute，代码停止重复采证并重分类。
- [x] 为 `expand_non_nested_family` 加入一阶 rate law 与二阶 kinematic force law；fresh V3.10通过，但变量名和完全观测shortcut必须在跨域包中继续攻击。
- [x] 分解 validation 语义：定位49-row/6-segment输入契约 bug，并修复 private case partition；fresh V3.9.1只授权 skeleton factorial。
- [ ] 将每种认识动作的成本、失败恢复和 stop condition 写入通用 action ledger；V3.8.1 已先为 acquisition 固化 cost=1、refute 与 stop receipt。
- [ ] 保持 task router 关闭，直到动作执行层在 fresh private WorldPack 中通过。
- [x] 大迭代19：冻结跨域 representation-shift WorldPack；匿名/置换/单位缩放状态，加入未见机制，并要求 topology hypothesis 不依赖语义变量名。

## 大迭代 20 候选

- [ ] 冻结至少两个不同 open-set 算子族，避免为 pendulum 单独硬编码 `sin` 后冒充通用演化。
- [ ] 实现 `ResidualSignature -> ConceptProposal -> typed expression grammar -> ConstantOptimizer`，任意代码执行保持关闭。
- [ ] 同预算比较固定目录与 residual-guided concept evolution；公开 evaluator 查询、表达式数、常数数和 wall-clock 全部记账。
- [ ] Experience buffer 保存全部成功/失败，不只 top-k；public score 不得进入 private admission。
- [ ] 用 accuracy/complexity Pareto、public OOD/perturbation challenge 和 fresh private WorldPack 决定 concept admit/reject/revoke。
- [ ] 设计内容寻址的 verified-lineage receipt，减少嵌套重放成本但不弱化 verifier 的 fail-closed 语义。

## 大迭代 21 候选

- [ ] 在 open-set 演化独立收口后，再单独引入部分观测、不规则采样和 latent-state/observability disposition；不得与大迭代 20 混改。
- [ ] 对无法识别的 observation model 输出 `NEEDS_STATE_RECONSTRUCTION`，不把预测拟合冒充机制恢复。
