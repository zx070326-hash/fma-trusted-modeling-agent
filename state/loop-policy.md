# 自主迭代策略

- 层级：当前仅 `Lv1 Manual`；不创建定时或无人值守任务。
- 权限：公开网络只读、本地工作区读写、本地测试；无外部写操作，无现实决策执行。
- 生成/评估：代码和提示层均分离。生成侧提出候选；评估侧掌握冻结切分、阈值和晋级规则，并默认输出失败或 `NEEDS_EVIDENCE`。
- 状态：最近事实写入 `state/log.md`，未完成项写入 `state/inbox.md`，阶段总结写入 `state/weekly-summary.md`；归档内容不自动载入。
- 停止：测试失败、证据绑定失败、预算耗尽、候选重复、不可识别或决策不稳定时，停止晋级并保留可重放工件。
- 升级：连续稳定、负例充分、全量回归无退化且人工批准后，才讨论 `Lv2` 调度。

## V4.0 实验性 Graph-Loop 策略

- 仍保持 `Lv1 Manual`，不创建定时或无人值守任务。
- 产品建模图与 Agent 开发图使用独立 `RunStore`；只通过脱敏、内容寻址、固定 source snapshot 的 receipt 连接。
- Model 只能生成候选；Harness 执行；Verifier 授予 `qualified`；仅 Human 可授予 `active` 或发布。
- evaluator 在一个 epoch 内冻结；修改 evaluator、阈值或 Harness 必须进入下一 epoch，并重跑 anchored regression、heldout 和负迁移检查。
- 开发 release 不授予科学有效性；产品经验不自动进入可信 epistemic graph。
- 当前 bridge reconciler 尚未实现，任何跨层 runtime release 只能保持 pending，不得自动部署。
