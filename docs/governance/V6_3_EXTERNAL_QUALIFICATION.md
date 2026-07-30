# FMA V6.3 外部预测资格协议与能力边界

## 1. 目标与非目标

V6.3 在 V6.2 本地科学闭环之上增加一个窄域、一次性、可重放的外部预测资格层。它回答的唯一问题是：

> 当前冻结的正值标量时间序列模型，能否在严格未见、独立测量的外部持出集上，通过预先冻结的预测指标与阈值？

V6.3 不重新解释 V6.2 工件，不把 S0–S6 工作流门等同于科学成功，也不授权现实行动。它目前只面向预测性 claim 和冻结的 aggregate normalized RMSE；不是通用科学资格、因果识别、机制确认、处方决策或部署安全协议。

## 2. 第一性原理成功层级

“代码运行成功”“预测有效”“决策有效”和“现实行动安全”是不同命题，必须逐层取得证据：

| 层级 | 要回答的问题 | V6.3 的位置 |
|---|---|---|
| 工程完整性 | 当前工件、哈希、阶段证书和重放是否一致？ | 必须先通过 V6.2 当前闭环重放 |
| 本地预测证据 | 注册适配器、L0–L4 和 leakage-safe rolling confirmation 是否通过？ | 外部资格的前置条件，不是外部资格 |
| 外部测量资格 | 数据是否严格未见、测量含义是否独立审查、外部环境是否真实独立？ | 由外部 custody 证明 |
| 外部预测资格 | 预注册预测在唯一一次外部聚合评估中是否满足冻结阈值？ | V6.3 能够授予的最高层级 |
| 前瞻决策有效性 | 使用预测作出行动，是否改善真实效用或 regret？ | 不在 V6.3 范围内 |
| 部署与行动安全 | 在线运行、监控、回滚和治理是否足以授权现实行动？ | 不在 V6.3 范围内 |

任何下层 PASS 都不能替代上层证据。外部预测资格只适用于“该任务、该模型身份、该外部快照、该指标、该阈值”的精确组合。

## 3. 资格状态机

```text
current V6.2 closure replay
  -> frozen qualification contract
  -> signed external custody + code-owned admission
  -> current-model prediction binding
  -> append-only external registry + S4 prediction seal
  -> pre-dispatch evaluation reservation
  -> one signed aggregate evaluation
  -> independent promotion
  -> read-only full-chain replay
```

### 3.1 Contract

Harness 首先从当前 `StageWorkspace` 重新推导 V6.2 local context，而不接受调用者自报的 PASS。只有以下条件同时成立，才允许冻结资格合同：

- 当前 S4、S6 gate 和 V6.2 closure 均可验证；
- stage evidence admission 与 executable candidate receipt 属于当前 attempt；
- 任务是非 fixture 的 predictive claim；
- 本地 adapter、rolling confirmation、source integrity 和 workflow integrity 均通过；
- 当前模型 ID、模型身份 hash、训练快照、科学 bundle 与预测输出 schema 被精确绑定。

合同在任何私有评估前冻结：指标公式、最大阈值、最小外部观测数、一次性评估预算、aggregate-only 反馈规则、角色密钥指纹和 coordinator/generator host 身份。结果可见后不得修改合同或换模型。

### 3.2 Custody

外部 custodian 对以下事实签名：

- 外部快照与 holdout commitment；
- 目标顺序与观测数量；
- normalization scale commitment；
- strict-unseen 检查；
- 独立测量审查；
- 外部环境验证；
- holdout 在预测前已冻结且私有值未披露。

Harness 对 custody 做代码级 admission。失败被持久化为 `REJECTED`；失败 custody 不得继续签发 prediction seal，也不得被“修文案”升级。

### 3.3 Prediction binding

预测必须来自当前 V6.2 已选模型，而不是调用者上传的任意数列。绑定至少覆盖：

- 当前模型身份、科学 bundle、训练快照和 executable receipt；
- 外部快照、目标顺序和预测向量；
- 真实 generator role execution receipt；
- 预测工件 hash、输出 schema hash；
- `private_holdout_targets_accessed=false`。

该绑定由本地 Harness authority 认证。V6.3 只验证这种绑定；面向真实运行的 forecast generator 和 binding issuer 仍需生产化实现。

### 3.4 Registry and seal

独立 registry 在访问 holdout 之前，把预测工件及完整上游 hash 链写入 append-only registry 并签名。Harness 验证：

- custody、registry 的主机与密钥相互独立；
- contract → custody → prediction binding → registration 的所有 hash 一致；
- 注册时间晚于合同和 custody，且早于任何私有评估；
- 注册的 candidate、training snapshot、external snapshot 和预测工件没有替换。

验证通过后，Harness 才能签发绑定当前 S4 gate 的不可变 prediction seal。

### 3.5 Pre-dispatch reservation

在把任务派发给 evaluator、尤其是在任何私有目标可能被读取之前，Harness 必须先写入唯一、不可变、经本地 authority 认证的 evaluation reservation。reservation 绑定：

- qualification、contract、custody admission；
- current-model prediction binding、registration、prediction seal；
- 预测工件；
- 唯一的 evaluator key、evaluator host；
- 固定的 evaluation sequence `1`；
- custodian、registry、coordinator、generator 的主机身份。

reservation 是“一次私测预算已经被预占”的证明。它必须先于 evaluator 输出，不能在看到分数后补写，也不能通过更换 evaluator 或时间戳获得第二次机会。

### 3.6 Aggregate evaluation

独立 evaluator 只能返回签名的聚合充分统计量和冻结指标：

- squared-error sum；
- target-squared-value sum；
- observation count；
- aggregate normalized RMSE；
- 与 contract、custody、registration、seal、reservation 的完整绑定。

协议禁止逐观测反馈、私有目标披露和第二次 evaluation。Harness 重算指标并以 create-once consumption 记录消耗；通过或失败都消耗这一资格任务的唯一评估机会。

### 3.7 Promotion

promotion authority 必须在独立主机上使用独立 Ed25519 key，对聚合结果、冻结阈值和 integrity incident 状态重新判断。只有以下条件全部成立，才可签署 `QUALIFY`：

- 外部评估满足冻结阈值与最小样本量；
- strict unseen、独立测量、外部环境和 aggregate-only 检查均成立；
- 所有签名、主机隔离、时间顺序和 hash 链通过；
- 没有 integrity incident；
- promotion 自身晚于 evaluation 且与完整链条一致。

缺少 promotion 时状态为 `NOT_RUN`；阈值失败或出现完整性问题时状态为 `REJECTED`。签名拒绝是已消耗评估后的最终科学结果，不能在同一 holdout 上调阈值重试。

### 3.8 Read-only replay

最终 verifier 必须只按 receipt 中的内容寻址 hash 重新加载全部 authority artifacts，重验签名、链式绑定、阈值、唯一消费记录和最终状态。

重放必须是纯读取：不得创建缺失工件、补 reservation、再次消费评估、重签 promotion 或改写 ledger。任何缺失、替换、过期或重算差异均返回 replay `FAIL`，但不得修改原资格结论。当前本地实现已经覆盖 reservation 与纯读取重放，并通过真实磁盘 `StageWorkspace` 重开测试；这仍只是协议实现证据，不是已经完成的真实外部运行。

## 4. 角色、主机与密钥边界

| 角色 | 职责 | 不得拥有的权力 |
|---|---|---|
| Coordinator / Harness | 重放 V6.2、冻结合同、验证绑定、签发 seal、预占评估、持久化 ledger | 不得看到私有目标、伪造外部签名或自行 promotion |
| Forecast generator | 用当前冻结模型生成目标有序的预测向量和执行 receipt | 不得读取 holdout target、换模型或签资格 |
| Custodian | 保管外部目标，证明 strict unseen、测量审查和环境独立性 | 不得生成预测、注册预测、评分后 promotion |
| Registry | 在私有访问前登记预测并签名 | 不得持有目标、评估或改写注册 |
| Evaluator | 在 reservation 后执行唯一一次 aggregate evaluation | 不得返回逐点反馈、泄露私有值或自行 promotion |
| Promotion authority | 独立复算阈值与完整性并签署 QUALIFY/REJECT | 不得修改合同、预测、评估或授权现实行动 |

合同固定 custody、registry、evaluator、promotion 四把外部 Ed25519 key 的 ID 与物理指纹；四者必须两两不同。Coordinator 与 generator 必须位于不同 host；custodian、registry、evaluator 必须彼此不同且不得与前两者重合；promotion host 还必须独立于所有前述 host。

当前代码中的 key fingerprint 和 host ID 检查是协议约束。它们尚不能替代真实外部 trust root、远程 host attestation、密钥生命周期管理和独立节点运维证据。

## 5. Claim ceiling 与行动边界

V6.3 的最终状态只有：

- `NOT_RUN`：链路缺件，例如没有独立 promotion；保留原本地 claim ceiling；
- `REJECTED`：外部阈值、资格条件或完整性检查失败；不得提升 claim；
- `EXTERNALLY_QUALIFIED`：只把上限提升到 `externally_qualified_predictive_evidence`。

即使获得外部预测资格：

- `mechanistic_qualification_granted=false`；
- `prescriptive_qualification_granted=false`；
- `real_world_action_authorized=false`。

因此不得把它表述为机制正确、因果有效、决策最优、系统安全或整个 Agent 已被通用认证。

## 6. 当前证据能证明什么

本地测试能够证明 schema 约束、hash/signature 绑定、角色隔离规则、失败关闭、一次性 reservation/consumption、阈值重算和 replay 逻辑在测试环境中按预期工作。fixture、fake workspace、同机进程和测试密钥都只是协议证据：

- 不证明真实外部资格已经运行；
- 不证明数据确实未见；
- 不证明测量审查真正独立；
- 不证明 host 或签名主体是真实第三方；
- 不证明模型已在外部环境泛化；
- 不授权任何现实行动。

## 7. 尚未关闭的真实能力缺口

1. 将已经通过真实磁盘重开测试的 V6.3 链接入真实 Studio 任务和未经替代的当前 V6.2 closure；现有集成测试仍固定了 closure summary 与 S4/S6 gate。
2. 真实 selected-model forecast runtime、role receipt、current-model binding issuer 以及预测/预约 single-writer coordinator candidate 已实现；仍须完成 custody、evaluation、promotion 的类型化 ingress、失败收据、人工 reconciliation 与受保护服务身份。
3. 建立外部 trust root、证书/密钥轮换、远程 host attestation 和可审计的独立运维边界。V6.4 当前只验证本地机制并强制输出 `NOT_RUN`，不能由本地自建 provider 抬升资格。
4. 在真实独立 custodian、registry、evaluator、promotion 节点上冻结并运行一个严格未见任务。
5. 将现有 create-once reservation 与 dispatch packet 接入真实 evaluator 调度，以 reservation hash 作为远端幂等键，并完成网络重试、超时、崩溃恢复与取消语义验证。
6. 另建 prospective decision trial、部署审批、在线监控、漂移检测、回滚和事故响应；这些不能由预测资格自动继承。

在上述证据产生前，V6.3 应被描述为“外部预测资格协议与本地实现候选”，而不是“已完成真实外部资格”。Coordinator 中的 V6.3 protocol flag 不等于科学资格；`scientific_qualification_granted` 保持为 `false`。
