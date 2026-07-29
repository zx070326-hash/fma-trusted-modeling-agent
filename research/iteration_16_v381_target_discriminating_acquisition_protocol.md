# 大迭代 16 / V3.8.1：Target-Discriminating Acquisition 与失败恢复协议

冻结时间：2026-07-22（private confirmation WorldPack 生成前）

## 1. 第一性原理问题

`NEEDS_EVIDENCE` 不是“继续采数据”的授权。一次认识动作只有在同预算对照下改善终端目标，才值得重复。否则 Agent 必须停止、保存失败证据并把问题重分类为估计器、验证语义或模型族不足。

V3.8 训练运行中，21 个 `acquire_target_discriminating_evidence` 全是 controlled-response target。离线训练诊断显示：随机追加一个安全输入或按三族轨迹分歧追加一个输入，目标条件模型覆盖均为 0/21。因此本轮不调阈值，冻结该候选策略，在全新种子上做前瞻反证尝试。

## 2. 单变量变化

保持不变：

- V3.7 的 linear/quadratic/cubic 三族、拟合器、`0.35` CV loss、rank/condition 门；
- V3.7 的 pilot + action `[0, 7]` 初始证据；
- V3.7.1 disposition policy；
- V3.8 target clarification、contract lineage 与目标条件语义；
- action catalog、输入幅度、成本、边界、噪声和 private target loss。

唯一新增：对 V3.8 输出 `acquire_target_discriminating_evidence` 的 case，执行一次同成本额外输入动作，并重做 target-conditioned challenge。

## 3. 双臂策略

- baseline：在剩余安全 catalog action 中，用 `sha256(case_id, policy_hash)` 确定性选择一个；
- candidate：用当前三条公开 observation 拟合 linear/quadratic/cubic 三模型，对剩余 action 做公开轨迹预测，选择归一化跨族轨迹方差最大的 admissible action；
- 两臂预算均为 1、动作成本均为 1；不能读取 private mechanism、private observation、probe 或 target loss 后再选动作；
- Harness 重新验证 action envelope，只有 `allow_synthetic` 后才能从 synthetic Reality Interface 取得 observation。

## 4. 重挑战

- controlled-response target：pilot 固定进入训练，对三条 action trajectory 做 leave-one-action-out；
- free-run target：三条 action trajectory 训练，pilot 作为唯一 held-out；
- family eligibility 继续使用 simulation、rank ratio `>=0.95`、condition `<=1e8`、target CV loss `<=0.35`；
- family selection 继续使用 one-standard-error 下最简单 family；
- 不改 cutoff，不以新增数据后的训练残差替代 held-out evidence。

## 5. Private adjudication

全新 16 seeds × 4 mechanisms；seeds 与 V3.7/V3.7.1/V3.8 均不相交。

对 source next-action 为 acquire 的 case：

- 两臂 action execution precision/recall 必须为 1；
- 非法/越权/重复动作必须为 0；
- unresolved case 的 adjudicated target loss 固定为 `10.0`；
- 计算 candidate 相对 baseline 的 resolved coverage、配对 mean loss improvement、2000 次 case bootstrap 95% CI；
- material negative transfer：candidate loss 比 baseline 高 `>0.1`；用一侧 95% Clopper-Pearson 上界；
- 不足 8 个目标 case 时自动失败，不追加样本。

## 6. 预冻结晋级门

只有全部满足才可进入一次性 confirmation：

1. 至少 8 个 acquisition target case；
2. 两臂全部按授权执行，零非法动作；
3. candidate resolved coverage 不低于 baseline；
4. paired mean target-loss improvement 的 bootstrap 下界严格大于 0；
5. material negative-transfer rate 一侧上界不超过 `0.10`；
6. evaluator 可从 sealed public state、private pack 和策略独立重放全部 proposal、execution、challenge、decision 与报告。

若失败，终态固定为 `target_discriminating_acquisition_refuted_v381`，恢复动作固定为 `stop_repeat_acquisition_reclassify_estimator_or_family`。失败后禁止继续重复该采证策略或放宽 CV 门。

若通过，终态只允许 `target_discriminating_acquisition_ready_for_confirmation_v381`。两种终态都不允许 qualification、confirmation、task router 或现实动作。
