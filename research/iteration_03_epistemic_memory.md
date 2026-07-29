# 大迭代 3：可撤销知识与方法学习证据记录

检索/访问日期：2026-07-22。只使用规范或方法原始来源来决定本轮最小实现；网页内容始终按不可信数据处理。

## 1. 第一性原理问题

一个长期运行的建模 Agent 如果会从网上和历史实验学习，就必须同时回答：

- 学到了什么精确主张；
- 来自哪个版本的来源和哪个实验；
- 这个主张适用于什么任务；
- 什么证据会反驳它；
- 来源被修订或撤销时，哪些下游结论必须失效；
- 和记忆关闭相比，它是否在未见任务上产生了成本归一化增益。

因此“向量检索 + 提示词”不是长期科学记忆的充分形式。最小充分结构是：内容寻址来源、类型化依赖图、独立隐藏评测、用途限定状态和追加式撤销。

## 2. 原始来源

### FPP3：Simple exponential smoothing

来源：[Forecasting: Principles and Practice, 3rd ed., §8.1](https://otexts.com/fpp3/ses.html)。

页面给出 SES 的水平递推：当前水平是新观测与上一水平的加权组合，`alpha` 位于 0 到 1；多步预测为同一最终水平。原文同时明确：这种平坦预测只适合没有趋势或季节成分的序列。因此本轮允许生成的不是“SES 普遍更好”，而是一个有适用条件和排除条件的候选演化算子。

本地捕获元数据：

- exact URL：`https://otexts.com/fpp3/ses.html`
- HTTP Last-Modified：`Sat, 18 Jul 2026 00:05:58 GMT`
- response hash：`b9222aeabddcd59b96627bda924f791d0da547cf2716e302f11687780a632dfe`
- 解释边界：`alpha=0.3` 是本地冻结的实验参数，不是来源推荐值；必须由隐藏评测裁决。

### W3C PROV-DM

来源：[W3C Recommendation: PROV-DM](https://www.w3.org/TR/prov-dm/)。

规范把 provenance 定义为产生数据或事物所涉及的实体、活动和责任信息，并提供 derivation、revision、invalidation 等关系。它还强调 derivation 路径可支持 provenance-based reproducibility，最新 URL 与特定版本 URL 的状态语义不同。

本轮只吸收三个原则，不声称实现完整 PROV：

1. 证据、主张和报告都是有固定版本的 entity；
2. 派生关系必须可追溯，revision 产生新 entity，不原位改写旧工件；
3. invalidation 是事件，历史记录保留，但当前有效状态必须改变。

## 3. 实现决定

| 来源事实或工程风险 | 实现决定 | 未解决边界 |
|---|---|---|
| 网页会更新且可能含指令文本 | exact URL allowlist、原始响应 hash、`untrusted_web_data` | 未做通用 HTML 语义抽取 |
| 方法有适用条件和排除条件 | `MethodClaimDraftV22` 显式记录二者 | 条件本身仍需经验验证 |
| provenance 需要 derivation/invalidation | 事件溯源 `EpistemicGraphStore` 与关系限定的撤销传播 | 没有远程不可变锚或签名 |
| 组合评测不能归因到单个组件 | 封存精确 `WorldPackArmPolicyV22` | 尚未做组件级 Shapley/因果归因 |
| 记忆复杂度必须挣得其成本 | 同预算、同 case 的有记忆/无记忆配对消融 | 12 case 对低负迁移率的统计把握不足 |

## 4. 证据分层结论

### 已有直接证据

- 本地 108 项回归覆盖图重放、篡改、环、语义状态、撤销 cascade、网页跳转/大小/media type、隐藏外层泄漏和 WorldPack 重算。
- 一个实际 FPP3 页面已被受控捕获并重放。
- 一个实际 BLS V2.2 影子运行被导入图；单独模拟撤销使 8 个下游节点失效。
- 12-case 消融显示 4 胜、8 平、0 负，平均改善 95% CI 为正，但事前胜率门未通过。

### 推断

- 事件溯源认识图比无版本向量记忆更适合科学知识复用，因为它能把来源修订传播到用途资格。
- “固定多个 Agent 人格”仍不是必要组件；本轮瓶颈来自证据对象、评测对象和门定义，而不是缺少更多对话角色。
- 当前记忆 policy 可能有安全 fallback 和局部迁移价值，但尚未获得正式晋级。

### 不确定性

- 0 个观察到的负迁移不能证明真实负迁移率为零。
- 合成 WorldPack 的机制覆盖和现实任务分布权重都有限。
- 网页来源正确支持 SES 公式与限制，但不能支持整个 memory policy 的效果；后者只由独立 WorldPack 评估。

## 5. 下一验证动作

冻结 V2.3 前瞻确认协议：使用全新 seed，增加 case 数；主指标改为宏平均配对增益的置信下界；安全指标使用负迁移率的一侧上置信界和逐机制非劣界；保持 policy、预算、选择器和外层完全冻结。旧 V2.2 报告永久保留 `candidate_rejected`，不能被新门追溯改写。

## 6. 后续执行结果

该动作已在大迭代 4–5 执行，原 V2.2 报告未改写：第一组 80 个新 case 因负迁移率上界 5.79% 被拒；失败驱动的新 safe policy 在第二组 80 个新 case 上得到宏平均改善 11.97%（95% CI `[11.01%, 13.11%]`）、0 个 >5% 负迁移（上界 3.68%），只获得 `synthetic_forecast_worldpack_v23` qualification。该后续结果不改变本文件中对网页证据、通用迁移和现实资格的限制。
