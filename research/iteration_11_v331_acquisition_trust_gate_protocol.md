# 大迭代 11：V3.3.1 acquisition trust gate 协议

冻结日期：2026-07-22（Asia/Shanghai）
运行等级：`L-DRAFT / Lv1 Manual`
父证据：V3.3 evolution hash `afe4ff296cdaa67f74c22a4e078377bdd1be7c41656ab9800498bb98cc7db0e1`

## V3.3 已冻结失败（不得回写）

V3.3 在全新 64-case pack 上实现逐 case 的 clarification entitlement、controlled-experiment entitlement、实际使用和目标契约 parity。资源混杂被排除后，未校准 V3.2 acquisition 的宏 absolute loss 改善为 `0.008278`，95% CI `[-0.044574, 0.060202]`；36 个 eligible case 有 7 次实质负迁移，一侧 95% 上界 `0.334393`；阻尼振子均值 `-0.019536`。终态 `acquisition_candidate_failed_v33`，无 qualification。

V3.3 中 case-level 平均 acquisition score 与真实 improvement 的 Spearman 相关约 `-0.0381`；score dispersion 和 top-two margin 也接近零相关。因此不允许直接把 posterior-risk proxy 当作收益置信度。

## 失败驱动的候选

先让两臂执行相同的两个预冻结随机输入。candidate 只在第三次实验前做一次 code-owned trust decision：

1. 用 public pilot + 第一个 anchor 拟合原样 V3.1 模型；
2. 在第二个、未参与拟合的 anchor 上计算 trajectory NRMSE；
3. 用已经执行的两个 anchor 更新 V3.2 acquisition；
4. 比较 V3.2 最优动作与共享随机 baseline 第三个动作的 robust goal-risk score；
5. 只有跨激励 NRMSE 和 score margin 同时通过，才用主动动作，否则执行完全相同的随机 fallback。

该结构把“模型是否值得信任”建立在未参与拟合的真实 synthetic observation 上，而不是训练残差、自报 posterior 或隐藏 probe 上。它仍不是校准概率或形式安全保证。

## 阈值冻结

在已经公开用于失败分析的 V3.3 pack 上，只检查以下离散网格：

- cross-excitation NRMSE：`[0.03, 0.04, 0.05, 0.06]`；
- goal-risk score margin：`[0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]`。

选择规则事先固定为：先要求 0 次 `>0.02` 实质负迁移，再最大化平均 improvement，再最大化主动切换数。所得候选为：

```text
maximum_cross_excitation_nrmse = 0.05
minimum_goal_risk_margin       = 0.03
anchor_experiment_count        = 2
```

在已见 V3.3 pack 上，该候选主动切换 12/36，0 次实质负迁移，平均 improvement `0.020860`；三个 in-family 机制均值分别约为 `0.016014 / 0.007703 / 0.038864`。这些只是候选生成数据，不能作为通过证据。

## V3.3.1 冻结边界

- 使用与 V3.3 相同的 typed resource ledger：两臂共享 clarification，且各有 3 次 controlled experiment；
- 两臂前两个实验动作逐 case 完全相同，第三个随机 fallback 也完全相同；
- 只改变 candidate acquisition 的信任与 fallback 规则；模型、动作目录、最终拟合器、风险门、统计门和 router 不变；
- trust receipt 必须绑定两个 anchor observation hash、cross-excitation NRMSE、active/fallback action hash、两者 score、margin、阈值和最终选择；
- trust gate 不得读取 hidden mechanism、hidden parameters、private probe、真实 target loss 或 expected route；
- 使用 16 个与 V3.1/V3.1.1/V3.2/V3.3 不重叠的新 seed；仍只允许 exploratory terminal status，不允许 confirmation 或 qualification。
