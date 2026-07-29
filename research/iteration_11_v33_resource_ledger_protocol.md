# 大迭代 11：V3.3 资源账本纯化协议

冻结日期：2026-07-22（Asia/Shanghai）
运行等级：`L-DRAFT / Lv1 Manual`
父证据：V3.2 evolution hash `99c7acd5d023741453566ac51da9ae8288fe757783b9124ca221fbb5011dc0e3`

## V3.2 已冻结失败（不得回写）

正式运行 `v32_goal_posterior_risk_exploratory` 使用 16 个新 seed、64 cases，其中 36 个 performance eligible。结果为：宏 absolute loss 改善 `2.298686`，95% 分层 bootstrap 区间 `[-0.041470, 6.843465]`；8 次实质负迁移，一侧 95% 上界 `0.365373`；router 保持 `0.75`。终态为 `acquisition_candidate_failed_v32`，无 qualification。

失败分层显示，8 次实质负迁移全部来自初始目标缺失的 27 个 eligible case；目标初始已知、两臂都执行 3 次实验的 9 个 case 为 0 次实质负迁移。旧 V3.1.1 启发式在同一已见 pack 上有 6 次实质负迁移；目标轨迹 sensitivity 线性化有 10 次；剩余预算 batch lookahead 有 6 次。有限动作目录的固定组合在 `(state_dimension, target)` 公开分层后仍无法消除二维 controlled-response 的负迁移。因此不把这些事后分析晋级为证据。

## 预冻结 V3.3：资源账本纯化实验

V3.2 的 scalar `action_budget=3` 把一次目标澄清和一次物理实验当成同一种资源：candidate 在目标缺失时只能执行 2 次输入，而 random baseline 执行 3 次。这使 acquisition 效果与 action debt 混杂。V3.3 只修复比较协议和资源表示：

- 两臂共享相同的目标澄清规则；目标缺失时都先绑定同一 synthetic value-owner evidence；
- 两臂都有 `clarification_budget=1` 与 `controlled_experiment_budget=3`，分别记账；
- 目标已知时两臂各做 3 次实验；目标缺失时两臂各澄清一次、再各做 3 次实验；
- baseline 仍是预冻结随机无放回，candidate 原样使用 V3.2 robust goal posterior-risk；
- estimator、动作目录、物理预算、经验风险门、router 和统计门全部不变；
- fresh exploratory seeds 与 V3.1/V3.1.1/V3.2 不重叠；V3.3 仍禁止 confirmation 和 qualification。

该设计比较的是“相同资源 entitlement 下的 acquisition”，不声称两种资源具有同一现实成本，也不证明澄清免费。现实成本需要独立 value owner 给出可比较效用后才能进入终端损失。
