# 大迭代 17 / V3.9.1：Evaluator Case-Partition Recovery

冻结时间：2026-07-22（V3.9.1 fresh private WorldPack 生成前）

## 事故

V3.9 fresh run 正确恢复了 384 个 action-fold input contracts，但 evaluator 用 private `performance_eligible` 划分质量 case。该字段同时把 12 个 sensor-calibration failure 和 16 个 Duffing model-mismatch sentinel 标为 false，导致：

- quality denominator 被错误写为 28；
- Duffing 从 private target-loss mechanism summary 中消失；
- `quality_abstention_preserved` 与 `skeleton_gap_remains_visible` 正确失败。

V3.9 运行保留为失败，不改写。

## V3.9.1 唯一变化

- quality partition 只能由三条公开 observation 的 `quality_flags` 重建；
- quality flags 非空的 12 case 必须双臂弃权；
- 其余 52 case 全部进入 coverage/private target-loss，包括16个 Duffing model-mismatch sentinel；
- 禁止读取 `performance_eligible` 决定 evaluator 分母。

模型、输入契约两臂、family、estimator、CV 门、selection、private loss、bootstrap、恢复门均与 V3.9 相同。

## Freshness 与终态

- 使用与 V3.7–V3.9 全部不相交的16个新 seeds；
- 只有 V3.9 原八个恢复门全部通过，才输出 `validator_input_contract_recovered_ready_for_skeleton_factorial_v391`；
- 该状态只恢复 validator，并保持 Duffing skeleton gap 可见；不允许 qualification、confirmation、task router 或现实动作。
