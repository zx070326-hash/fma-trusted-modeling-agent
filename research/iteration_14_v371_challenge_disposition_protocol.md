# V3.7.1：Challenge Disposition Controller 前瞻协议

## 冻结原因

V3.7 正式运行 `v37_applicability_model_challenge` 的终态为 `model_challenge_failed_v37`。52 个数据质量通过 case 中，candidate 只产生 25 个模型；27 个 `NEEDS_EVIDENCE` 按预注册 loss 10 进入 private evaluator。宏 improvement 为 `-5.38566`，95% CI `[-6.01895, -4.78367]`。

失败并非单一阈值偏紧：

- 12 个 exponential case 三个 family 的 held-out mean loss 全部超过 `0.35`；
- logistic 只有 2/12 case 产生 eligible family；
- damped 只有 7/12；
- Duffing 16/16 选择 linear，虽然 private target loss 相对 fixed quadratic 平均改善 `0.30540`，但 family routing 不正确；
- 许多公开 contract 的 `target_status` 仍是 `default_unverified`。它们虽然携带一个具体默认 target，却明确声明 `unresolved_fields=[decision_target]`；V3.7 漏掉了这个 authority 状态并直接进入 family selection。

因此 V3.7 的首要缺口是：它能给出挑战失败，却没有把失败变成下一步认识动作。V3.7.1 不改变 family、观测、CV、selection、private loss 或效果门，只新增一个类型化 disposition controller。

## 训练诊断与前瞻边界

V3.7 失败运行只用于冻结一个模型层诊断：

```text
nonlinear_residual_gain =
  (linear derivative residual - min(quadratic, cubic derivative residual))
  / max(linear derivative residual, 1e-12)
```

V3.7 中 Duffing 的该量约为 20%–30%，其它机制总体远低于此。因此预冻结触发值为 `0.15`。它只是经验 failure-signature 阈值，不是概率校准、结构辨识证明或普适 domain rule。

正式 V3.7.1 使用 16 个全新 seed；不得回看其 private outcome 修改阈值或优先级。

## 冻结优先级

Disposition 只能读取 V3.7 的公开 `ApplicabilityState`、`FamilyChallengeReceipt` 和 `ModelChallengeDecision`：

1. 出现 quality flag → `data_layer / repair_data_quality`；
2. contract 的 `target_status=default_unverified` 或仍含 `unresolved_fields` → `problem_layer / clarify_decision_target`；
3. 无 selected family → `data_layer / acquire_target_discriminating_evidence`；
4. 已选择 simple family，但 `nonlinear_residual_gain >= 0.15` → `model_layer / expand_non_nested_family`；
5. 否则 → `evaluation_layer / proceed_private_validation`。

所有 disposition 都是草稿动作：`execution_permitted=false`。V3.7.1 不执行澄清、采样、family 扩展或 private promotion。

## 私有 evaluator

隐藏 evaluator 按同一优先级构造 oracle disposition；只有第 4 条使用隐藏机制：Duffing 且没有选择 cubic 时期待 `expand_non_nested_family`。报告以下指标：

- 64 case 全覆盖；
- 总 route accuracy `>=0.90`；
- 每个实际出现的 oracle action accuracy `>=0.80`；
- 不得在 oracle 要求修复、澄清、采证或扩 family 时错误 `proceed_private_validation`；
- disposition、state、challenge、decision 和 source V3.7 failure hash 全部内容绑定；
- 新增 target-aware state 必须绑定公开 contract hash、target status 和 unresolved fields；evaluator 直接从冻结公开 contract 重算 oracle，不能复用 generator 的 target-state 判断；
- 无 private mechanism/probe/loss 字段进入 disposition；
- private pack 再生成、V3.7 两臂重放、V3.7 report 复算、V3.7.1 disposition/report 复算和事件链全部通过。

全部通过只产生：

```text
challenge_disposition_ready_for_synthetic_action_experiment_v371
```

它仅允许下一轮构造受限合成认识动作实验。`task_router`、qualification、confirmation 和 real-world authorization 仍全部为 false。
