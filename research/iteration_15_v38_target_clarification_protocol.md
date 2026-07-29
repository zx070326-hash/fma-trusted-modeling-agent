# 大迭代 15 / V3.8：Target Clarification Action 协议

## 目的

V3.7.1 在 fresh 64 cases 中把 39 个 case 路由为 `clarify_decision_target`。V3.8 执行这一认识动作，并检验它是否真正改变后续 challenge，而不是只生成一个正确标签。

本轮保持 `L-DRAFT / Lv1 Manual`。clarification source 是 synthetic value owner；没有现实用户、设备或外部系统动作。

## 训练诊断与不允许的外推

在已封存 V3.7.1 运行上进行的训练诊断：39 个 clarification case 中，冻结默认 target 对 26 个 controlled-response case 是错的；澄清后 target 全部可恢复。按 target 重新选择 held-out folds 后，free-run case 往往获得可用模型，controlled-response case 则经常转为 `acquire_target_discriminating_evidence`。

这不证明 clarification 改善总体 target loss。弃权/采证的效用需要 context-of-use 的 abstention cost，当前不存在该权威代价，因此 V3.8 不把弃权随意换算为效用，也不以总体 loss 晋级。

## 冻结输入

- 16 个全新 seed × 4 mechanisms；
- V3.7 family、观测、CV folds、rank/condition/loss thresholds 和 V3.7.1 disposition priority 全部不变；
- clarification policy 精确绑定通过验证的 V3.7.1 report；
- 只有 source disposition 为 `clarify_decision_target` 的 case 可执行；其它 case 必须 `not_applicable`；
- 每个 action budget 为 1，执行后为 0。

## Clarification action

执行 case 必须产生：

1. `TargetClarificationEvidenceV31`，来源 `synthetic_value_owner:<case_id>`；
2. V2 authoritative contract：parent hash 等于原 contract，triggering evidence hash 精确绑定 clarification evidence；
3. target-aware applicability state：`target_status=authoritative`、`unresolved_fields=[]`；
4. target-conditioned family challenges：
   - `free_run_prediction` 只使用 pilot holdout fold `[0]`；
   - `controlled_response_prediction` 使用两个 controlled holdout folds `[1, 2]`；
5. 不变的 rank/condition/CV threshold 与 one-standard-error selection；
6. 新的 typed next disposition：`acquire_target_discriminating_evidence`、`expand_non_nested_family` 或 `proceed_private_validation`。

## 独立评价

private evaluator 直接从 frozen private case 的 true target 和原 public contract 重建：

- 应执行 clarification 的精确 case 集；
- before/after target accuracy；
- evidence 与 contract lineage；
- target-conditioned fold 索引、均值、标准误、eligibility 和 family decision；
- 非 clarification case 的零执行；
- selected-model private target loss 只作为分目标诊断，不作为 qualification gate。

预冻结 gates：

- action precision/recall 均为 1；
- after target accuracy = 1 且严格高于 before；
- 所有 executed contract authority/lineage/evidence 一致；
- 所有 conditioned challenge/decision 可独立复算；
- 至少出现一个 `acquire_target_discriminating_evidence`，防止 clarification 被误报为终局；
- 无 task router、qualification、confirmation 或 real-world authorization。

通过状态仅为：

```text
target_clarification_ready_for_composed_synthetic_loop_v38
```

它允许下一轮把 clarification 与 acquisition 组合进 synthetic loop，不授权模型部署或现实动作。
