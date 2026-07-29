# FMA 自主迭代日志

本文件保存最近迭代的热状态。记录只陈述已经运行或已经验证的事实；提案、推断与能力声明分开。

## 2026-07-22 — 大迭代 1（已完成，目标整体仍进行中）

- 目标：补齐首个经验建模闭环：结构化历史数据、非同构候选、滚动时间持出、不确定性、容量决策稳定性与代码级弃权。
- 风险等级：`L-DRAFT`。只允许读取公开网络资料、修改本地代码/文档、运行本地实验；禁止现实业务动作。
- 运行方式：`Lv1 Manual`。尚未证明连续 30 天稳定，不进入无人值守定时执行。
- 基线：`python -m pytest`，结果 `79 passed in 27.09s`。
- 已证能力：使命/审批绑定、问题发现草稿与 admission、公开提案/私有测试隔离、确定性 bounded-ILP 验证、内容寻址与哈希链、fixture/fake-CLI 回归。
- 未证能力：真实历史数据经验有效性、可迁移 UQ、决策用途资格、真实 Codex 建模质量、开放世界自主建模。
- 独立评估规则：候选生成模块不能决定自身通过；经验 evaluator 默认失败，只有冻结数据、冻结时间切分和冻结阈值全部满足时才能晋级。
- 实现：`TimeSeriesDataContract/Snapshot`、严格本地 CSV intake、四模型 portfolio、rolling-origin evaluator、经验残差区间、容量 regret/行动一致性门、内容寻址 manifest 与全链复算。
- 稳定 fixture：4/4 候选通过，容量均为 10，终态 `decision_eligible@synthetic_fixture_analysis`，不授权现实动作。
- 结构突变 fixture：仅 last-value 通过预测门且 regret 超限，终态 `NEEDS_EVIDENCE`。
- 负例：未来值泄漏、重封伪报告、重复时间戳、工件篡改均被测试覆盖。
- 最终回归：86 个测试通过；两个落盘实验在最终代码下复算验证均为真。
- 下一主要缺口：真实公开数据 shadow、漂移/OOD、敏感性/可识别性、跨任务家族评测与置信界。

## 2026-07-22 — 大迭代 2（已完成，目标整体仍进行中）

- 官方来源：BLS 两个 2016–2025 月度 series；USGS 两个 2023–2024 日度 site。所有请求 allowlist、来源身份、连续频率、原始响应与 revision status 均冻结。
- 统计门：同一 rolling holdout 上的配对 circular moving-block bootstrap；3 seeds × 500 replicates；报告效应量、95% 区间与方向，不用 p 值替代效果。
- 探索结果：默认 global mean/trend/naive seasonal 在两个真实域均显著差于 last-value，正确 `NEEDS_EVIDENCE`。
- 演化：冻结 local mean、local trend、exponential smoothing 和年度水文季节算子，转到未见 series/site 确认。
- 确认结果：BLS local trend 与 exponential level 的 MAE 改善区间严格大于 0，但漂移门阻止总晋级；USGS 全部 challenger 仍差于基线。
- 设计拒绝：无权威水文阈值/损失/context of use，因此不伪造阈值决策 gate。
- 兼容事故：新增字段曾污染 V2.1 hash；重放发现后恢复旧 schema，新建 V2.2 方言，保留并标记 superseded 运行，加入 known-hash 回归并重跑。
- 最终证据：2 个探索 V2.1 + 2 个确认 V2.2 真实运行均可从原始 JSON 全链复算；全量 95 个测试通过；现实动作始终未授权。

## 2026-07-22 — 大迭代 3（已完成，目标整体仍进行中）

- 目标：建立可撤销长期科学记忆，并用 private-outer 同预算消融检验记忆是否真正帮助未见任务。
- 实现：事件溯源 `EpistemicGraph V2.2`、typed dependency/refute/supersede、effective status、关系限定的 cascade、exact-URL 方法网页快照、候选知识 admission、精确 `WorldPackArmPolicyV22`。
- 在线证据：实际捕获 FPP3 SES 页面，response hash `b9222aea...32dfe`；网页只形成 `candidate_requires_hidden_validation`，不直接晋级。
- 撤销证据：实际 BLS V2.2 运行导入图；单独模拟 revision drill 从 raw source 撤销并级联 8 个下游节点；主图未被模拟演练污染。
- 隐藏消融：4 机制 × 3 seed；两臂均 4 候选、每例 96 次内层评估、24 点 private outer。memory 4 胜/8 平/0 负迁移，平均 MAE 改善 95% CI `[0.0153, 0.1097]`。
- 事前裁决：严格胜例比例 4/12 未达到 50%，因此终态保持 `candidate_rejected`；没有事后修改门或追溯晋级。
- 认识修复：组合级 WorldPack 报告只能裁决精确 ArmPolicy，不能错误归因给单个 SES 组件；该绑定已机械化。
- 最终证据：主认识图 18 节点/18 边可重放；全量 108 测试通过；现实动作仍未授权。
- 下一缺口：用全新 seed 和更符合“正期望 + 非劣 + 低负迁移”的预注册 V2.3 门做前瞻确认。

## 2026-07-22 — 大迭代 4（已完成，未晋级）

- V2.2 旧报告未改写；V2.3 spec/policies 在 private pack 之前写入哈希事件链。
- 20 个全新 seed × 4 机制 = 80 case；同 4 候选、同 96 次内层评估。
- 宏平均相对改善 12.70%，95% CI `[11.29%, 14.34%]`；四机制均满足 5% 非劣。
- 发现 1/80 个 >5% 负迁移；一侧 95% 上界 5.79% 超过 5% 安全线，终态 `candidate_rejected_v23`。
- 没有追加样本压上界；失败签名是 memory 丢失 global trend、误选 SES。

## 2026-07-22 — 大迭代 5（已完成，获得窄 qualification）

- exact policy 演化为 `last + global trend + local trend + seasonal`；候选/评估预算不变。SES 知识保留，但从该 policy 退役。
- 第二组 20 个全新 seed × 4 机制 = 80 case；沿用未改变的 V2.3 统计/安全门。
- 宏平均相对改善 11.97%，95% CI `[11.01%, 13.11%]`；四机制 5% 非劣；0/80 >5% 负迁移，一侧 95% 上界 3.68%。
- 终态 `promoted_for_worldpack_scope_v23`；qualification 仅限 `synthetic_forecast_worldpack_v23`。
- 主认识图更新为 28 节点/28 边；旧 policy 为 `refuted`，新 policy/qualification 为 `active`，依赖可撤销。
- 最终全量回归：114 项通过。
- 下一步必须跨到不同 IR/机制家族，不能把单一合成预测家族的通过外推成通用建模能力。

## 2026-07-22 — 大迭代 6（已完成，Dynamics policy 未晋级）

- 在线证据：冻结 SINDy、Weak SINDy、observability/structural-identifiability 三个精确 DOI/Crossref 响应；只形成 `candidate_requires_hidden_dynamics_validation`。
- 新方言：`DynamicsDataSnapshotV24`、显式多项式 basis、Savitzky–Golay 导数、ridge/STLSQ、内容寻址 `DynamicsModelIRV24`、独立积分和轨迹条件 rank/condition 诊断。
- 边界：rank 诊断固定 `structural_identifiability_proven=false`；部分观测和未激发 sentinel 均正确弃权。
- 探索 24 case：sparse memory 17 胜/2 平/5 负，结构 F1 改善区间为正，但 3 次严重负迁移且 Lotka–Volterra cubic 外推灾难，终态 `exploratory_only`。
- 第一次 80-case 确认：删除 cubic 并保留 dense fallback；宏改善 15.22%（95% CI `[9.93%, 20.46%]`），但阻尼振子非劣失败、14 次严重负迁移，拒绝。
- 第二次 80-case 确认：增加 dense guard 与 10% inner safety margin；结构 F1 改善 7.76%（95% CI `[4.69%, 10.85%]`），但轨迹宏改善 0.30%（95% CI `[-9.41%, 8.67%]`）、9 次严重负迁移，仍拒绝。
- 主认识图更新为 53 节点/53 边；两个确认 policy 均为 `refuted`，候选知识仍 active，不存在 Dynamics qualification。
- 最终全量回归：124 项测试通过；knowledge、探索、两次确认和主图均可独立重放。
- 下一缺口：weak/integral matching、参数 UQ/稳定性约束、主动激励与真正的 observability/structural-identifiability 工具。

## 2026-07-22 — 大迭代 7（已完成，积分估计器未晋级）

- 证据边界：原始 Weak SINDy 论文确认 `phi=1` 只是积分方程特例；实现命名为 `window_integral_matching`，不声称完整 WSINDy。
- 加法 V2.5：point/integral typed policy、fit/model/selection、单组件机械检查、协议先冻结、private WorldPack、manifest 与独立重放；V2.4 历史 hash 未改。
- 探索 1：15 点积分窗 32 cases 中 9 胜/23 负、21 次实质负迁移，宏改善 -127.94%（95% CI `[-210.92%, -56.83%]`）。
- 失败演化：只改为 41 点窗/步长 5；新的 32 cases 为 16 胜/16 负，宏改善 -2.84%（95% CI `[-17.34%, 11.49%]`），仍不够晋级。
- 确认：80 个全新 cases 为 38 胜/42 负、39 次实质负迁移；宏改善 -31.08%（95% CI `[-56.99%, -8.79%]`），四机制非劣全部失败；结构 F1 仍改善 6.15 个百分点。
- 稳定性：冻结 200 次 moving-block 协议；integral 23/80 case 不稳定，高于 10% 门。该结果是依赖感知稳定性诊断，不是校准置信区间。
- 终态：`candidate_rejected_estimator_v25`；无 qualification；主认识图 75 节点/84 边，确认 integral policy 为 `refuted`，snapshot `6a2db5c4a4e196be28f5600cc40aa1e36c84183936b4eff4e46e3b7461ad99b8`。
- 下一步：主动激励/可控输入、信息矩阵与按机制实验设计；停止继续扫积分窗。
- 最终机械验证：130 项测试通过；三条 V2.5、三条历史 V2.4 和 75 节点/84 边主图均独立重放通过。

## 2026-07-22 — 大迭代 8（已完成，获得窄主动实验 qualification）

- 第一性原理转向：V2.5 说明同一被动轨迹上更换估计器不足；V2.6 把安全初值 reset 作为有限实验动作，建立 propose → validate → execute → observe → update 的顺序循环。
- 权责：policy 只看到公开 pilot、已观测轨迹和动作目录；Harness 拥有动作边界、无重复、预算、隐藏模拟器、probe、报告、事件链和 qualification。
- 同预算消融：`random_safe_catalog` 对比 `ensemble_disagreement_catalog`；每例共同 pilot + 4 动作，每动作 25 点，拟合器、噪声、候选库和 4 条隐藏 probe 完全相同。
- 探索 1：32 cases 的原始相对效应为 -2.0945（95% CI `[-4.8771, -0.1118]`）；发现近零 baseline loss 的分母奇异性，不把它误判为动作策略证据。
- 单组件演化：只新增 V2.6.1 联合损失绝对差和 0.01 实际退化阈值；第二组不相交 32 cases 改善 0.03594（95% CI `[0.01856, 0.05479]`），0 次实际负迁移。
- 确认：80 个新 cases 改善 0.03430（95% CI `[0.01580, 0.05496]`）；5 次实际负迁移，上界 12.69%；四机制均非劣，宏 log-condition 改善 0.24534，非法动作 0。
- 终态：`promoted_for_synthetic_active_design_worldpack_v261`；qualification 仅限 `synthetic_safe_initial_condition_design_v261`，明确 continuous_control/real_world_validity/structural_identifiability 均为 false。
- 认识图：97 节点/112 边，qualification active，snapshot `57f75e03860cf4d1a74e63a10d9a7bfe7f4e0d055ef4494e7eaee7b249e5023e`。
- 限制：Lotka–Volterra 贡献最大，logistic 条件数诊断略退化；下一轮需效用尺度敏感性、风险/成本前移和有界连续输入方言。
- 最终机械验证：133 项测试通过；三条 V2.4、三条 V2.5、三条 V2.6/V2.6.1 运行与主认识图全部独立重放通过。

## 2026-07-22 — 大迭代 9（已完成，获得窄认识闭环 qualification）

- 架构纠偏：现有链最强在候选后的可信验证，最弱在固定问题之前和失败回溯之后；V3 改为稳定 `MissionConstitution` 加不可变但可被证据 supersede 的 episode 契约。
- 认识动作：策略可在采数据和澄清损失语义之间提议；Harness 独占类型、权限、成本、synthetic Reality Interface、父子契约绑定、shadow decision 和 private adjudication。
- V3.0 一步探索：32 cases 宏 regret 改善 0.08643（95% CI `[0.05028, 0.12415]`），但 2 次实质负迁移、上界 18.39%；旧结果保留为 failure signature。
- 单组件演化：V3.0.1 只把 action horizon 从 1 改为 2；baseline 采两批，candidate 缺失时先澄清再采一批，同总成本。
- V3.0.1 探索：32 cases 宏改善 0.10651（95% CI `[0.06829, 0.14681]`），0 次实质负迁移；样本上界 8.94%，未提前晋级。
- 确认：80 个全新 cases 宏改善 0.09770（95% CI `[0.07494, 0.12175]`），四机制均正，0 次实质负迁移、上界 3.68%；60 个缺失语义全部证据绑定重构，20 个已知语义零误改。
- 终态：`promoted_for_synthetic_epistemic_loop_v301`；qualification 仅限 `synthetic_sequential_problem_reformulation_capacity_worldpack_v301`，开放问题发现/真实有效性/现实动作/广义建模均为 false。
- 认识图：117 节点/132 边，V3.0 candidate refuted，V3.0.1 qualification active，snapshot `f5be3a86b313ec2091948227e611a7814a0c38e293a399d4f6db57acc0634d7d`。
- 最终机械验证：142 项测试通过；V2.4/V2.5/V2.6 系列 9 条历史运行、V3 三条新运行和 117/132 主认识图全部独立重放通过；CLI fixture smoke 也返回经事件链验证的结构化结果。

## 2026-07-22 — 大迭代 10（已完成协议与失败演化，无 qualification）

- 第一性原理定位：FMA 是可信证据/权限内核，不是终局；目标形态是以终端决策损失为目标的受治理认识控制系统。
- 原始来源：L4DC 2025 informative input design、JMLR 2022 nonlinear active identification、COLT 2020 linear active identification、OED/goal-oriented OED、PMLR 2021 safe learning。论文的线性/特征嵌入/一致模型集合保证均未外推。
- V3.1 typed IR：已知 `B`、六段输入、峰值/能量/切换/成本、D-opt/分歧/目标信息、经验风险、allow/deny/abstain、合成观察和内容哈希。
- 跨层 WorldPack：32 cases；18 性能 eligible、6 数据校准 sentinel、8 Duffing 模型错配 sentinel；24 个问题语义缺失。
- V3.1 结果：宏改善 -0.04786（CI [-0.13865, 0.00472]），damped -0.18524，6/18 实质负迁移；路由 75%；数据门、问题重构、非法动作和隐藏越界门正确。
- V3.1.1 单组件演化：只把 horizon 2→3；宏均值 +0.10191，damped +0.25928，但 CI 下界 -0.02469、仍 6/18 负迁移、router 仍 75%。无 qualification。
- 两条运行均在当前代码独立生成/执行/复算通过：`v31_exploratory_failure`、`v311_horizon_evolution_failure`。
- 新增 8 项专项测试，结果 `8 passed in 48.05s`；最终全量回归 `150 passed in 129.28s`。
- 下一失败签名：启发式 acquisition 不是终端 probe loss 的可靠代理；Duffing 局部二次训练残差不能作为模型错配检测。

## 2026-07-22 — 大迭代 11–13（已完成，受控动力学分支冻结）

- V3.2–V3.3.2 依次加入目标后验风险、资源账本、trust gate 与 paired-advantage；关键正式实验仍未约束跨机制负迁移。
- V3.4 单次失配中断伤害目标，V3.4.1 两次持续失配仅窄通过 interaction gate；只授权 acquisition retest。
- V3.5 三臂 factorial 将 selector 主效应与 Runtime Adapter moderation 分离；V3.6 的历史 q20 outcome cutoff 在 held-out 包中反而削弱效果。
- controlled-dynamics acquisition 分支冻结为 `refuted_pending_new_state_representation`；全量回归为 186/186。

## 2026-07-22 — 大迭代 14（已完成，获得窄 failure-to-action readiness）

- 在线证据：模型判别、模型 discrepancy、held-out predictive comparison 与适用输入区域四组原始来源；所有理论保证均未迁移。
- V3.7 typed applicability state + linear/quadratic/cubic challenge：52 个质量合格 case 只覆盖 25 个，宏 improvement -5.38566，正式失败。
- 失败说明 challenge gate 只能说“不”，不能决定下一认识动作；新增 V3.7.1 disposition controller，不改 V3.7 模型、数据、阈值或 private evaluator。
- 首个 V3.7.1 运行因 state 未保存 `target_status`，generator/evaluator 共享盲点而虚假 100%；该工件保留且当前 verifier 返回 false。
- 修正版绑定 public contract authority state，并让 evaluator 独立从 contract 重建 oracle；fresh 64 cases 路由为修数据 12、澄清目标 39、采证 7、扩 family 4、private validation 2，五类 accuracy 均 1.0。
- 终态仅为 `challenge_disposition_ready_for_synthetic_action_experiment_v371`；模型挑战仍失败，无 task router、qualification、confirmation 或现实授权。
- V3.7/V3.7.1 专项各 5 项通过；全量回归 `196 collected / 196 passed`，`983.5s`。

## 2026-07-22 — 大迭代 15（V3.8 目标澄清动作窄通过）

- 执行 39/39 个 `clarify_decision_target`；每次动作成本 1，仅允许 synthetic value-owner evidence，现实执行和 task router 均关闭。
- 目标状态由 `default_unverified` 经证据绑定 V2 contract 更新为 `authoritative`；evaluator 从 private WorldPack 独立重建 true target 与 lineage。
- target accuracy 从 `33.33%` 提升到 `100%`，action precision/recall 均为 `1.0`。
- 澄清后 21 个案例仍需 target-discriminating evidence，9 个需非嵌套 family，9 个可进 private validation；这明确反证“澄清目标就等于完成建模”。
- 终态仅为 `target_clarification_ready_for_composed_synthetic_loop_v38`；无模型 qualification、confirmation、task-router 或现实授权。
- V3.8 专项 `5 passed`；全量回归 `201 collected / 201 passed`，退出码 `0`，墙钟 `1043.8s`。

## 2026-07-22 — 大迭代 16（V3.8.1 判别采证被前瞻反证）

- 第一性原理修正：`NEEDS_EVIDENCE` 不是采数据授权；认识动作必须在同预算 private outer 下证明边际价值。
- 训练源 21 case：random/disagreement 各追加一次 action 后均 0 resolved；failure evidence hash `040d55df...0f67e`。
- fresh 22 target cases：两臂各执行 22 次，零非法动作，但 resolved coverage 均为 0，配对 improvement `0`，CI `[0,0]`。
- 终态 `target_discriminating_acquisition_refuted_v381`；代码恢复动作 `stop_repeat_acquisition_reclassify_estimator_or_family`。
- verifier 首次因不存在的 storage API fail closed；改为内容寻址加载后，同一未改写运行完整重放为真。
- V3.8.1 专项 `5 passed`；全量回归 `206 collected / 206 passed`，退出码 `0`，墙钟 `1129.2s`。

## 2026-07-22 — 大迭代 17（V3.9/V3.9.1 validator/evaluator 语义恢复）

- 根因：49行逐时刻 observation inputs 被传给6段输入 simulator；generator/evaluator/verifier 共享同一错误，所以机械重放无法发现科学语义失真。
- training 22 cases：完整三折 legacy 8 resolved，action-hash recovered 22 resolved；但 recovered Duffing private loss `1.05843`，骨架 gap 保留。
- V3.9 fresh 输入契约恢复，但 evaluator 用 private `performance_eligible` 删除 Duffing，终态失败并保留。
- V3.9.1 fresh 只用公开 quality flags：52 effect/12 quality，coverage `40.38% -> 100%`，paired improvement CI `[4.395, 7.249]`，Duffing loss `1.10359`。
- 终态仅 `ready_for_skeleton_factorial_v391`；V3.7/V3.7.1/V3.8.1相关数值归因 superseded，无 qualification 或现实授权。
- 专项 `5 passed`；全量回归 `211 collected / 211 passed`，退出码 `0`，墙钟 `1220.8s`。

## 2026-07-22 — 大迭代 18（V3.10 状态拓扑骨架因子实验通过）

- 旧 linear/quadratic/cubic 被重分类为嵌套容量，不再冒充完整 skeleton diversity；新增一阶 rate law 与二阶 kinematic force law。
- 正交冻结 skeleton × estimator × validator：5×2×2；每个性能case实际4个兼容骨架、16 cells、8 pairs。
- 开发阶段先修复 NumPy `trapz` 兼容失败；再发现parsimony在统计等价时强制切换造成1次材料性负迁移，新增只使用公开LOO的一标准误 switch guard。
- fresh 48 cases：39 performance/9 public-quality，`1872/1872` semantic input bindings；coverage均为1。
- private mean loss `0.352936 -> 0.016675`，paired improvement `0.336261`，95% CI `[0.192755,0.495582]`。
- Duffing `1.045960 -> 0.004451`；0/39材料性负迁移，一侧95%上界`7.394%`；全部11门通过。
- 终态仅 `skeleton_factorial_ready_for_cross_domain_v310`；变量名/完全观测仍是fixture shortcut，无qualification、task router或现实权限。
- 专项 `5 passed`；全量回归 `216 collected / 216 passed`，退出码`0`，墙钟`1342.1s`。

## 2026-07-22 — 大迭代 19（V3.11 表示不变拓扑确认通过）

- 构造匿名 `z0/z1/z2` public pack 与隐藏 mechanism/representation/pair/coordinate transform/OOD probe 的 private pack；Generator 的执行签名不接受 private pack。
- 用 thermal relaxation、Van der Pol、Lotka–Volterra、SIR 四类受支持机制与 pendulum open-set sentinel，成对检验 reference 与 scaled/permuted 表示。
- 第一次开发包内容正确但审计时间误写到未来；确认包尚未生成即停止。新增 lineage 时间单调和未来墙钟门，恢复版开发包独立重放通过；正式确认只绑定恢复版报告 `28e43334...d12b`。
- fresh confirmation：70 performance/10 public-quality；受支持 coverage、topology accuracy、open-set abstention、pair topology consistency 均为 1.0。
- private loss `5.443685 -> 2.001436`，paired improvement CI `[2.309513,4.581231]`；最大表示对 loss 差 `0.004213`；0/70 材料性负迁移，上界 `4.189%`。
- 13/13 冻结门通过，终态仅 `representation_topology_confirmed_v311`；无 qualification、task router 或现实权限。
- 专项 `5/5 passed`（`265.8s`）；全量回归 `221 collected / 221 passed`，退出码 `0`，墙钟 `1611.9s`。
- 下一缺口：V3.11 只会选择/拒绝冻结 topology；大迭代 20 转向 residual-guided concept proposal、受限 expression grammar、Pareto/OOD challenge 和 private concept admission/revocation，至少两个 open-set 算子族。
