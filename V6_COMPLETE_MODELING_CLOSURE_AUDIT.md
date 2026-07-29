# FMA 完整数学建模闭环审计

## 结论

FMA 当前已经是一个可信、可恢复、可审计的**窄域预测建模纵切**，但还不是能独立完成任意真实世界或前沿数学建模问题的通用 Agent。

当前最强的可执行能力是：

1. 从冻结问题契约出发进行多分支 S1 探索、有限知识共享和独立形式化审计；
2. 对正值标量时间序列运行已注册的自治 ODE 或 adaptive positive-series 建模链；
3. 完成来源/变换绑定、候选搜索、L0--L4、rolling-origin confirmation、UQ、报告和论文投影；
4. 失败后在图中撤销下游证据，执行有预算的 retry、patch、branch、acquire-data、abstain 或 human 路由；
5. 把“工作流跑通”“本地预测证据”“外部预测资格”“决策价值”“机制解释”“现实行动”分成不同的证据层。

它仍不能诚实声称：

- 任意模型、任意数据、任意领域均可自动建模；
- 局部拟合或阶段 Gate 等于科学成功；
- 同一主机生成的多把密钥等于外部独立复现；
- 预测成功等于机制正确、决策有效或部署安全；
- 当前测试已经完成真实外部未见资格实验。

## 第一性原理：什么叫“建模成功”

建模成功不是单一布尔值，而是一个相对于**问题、claim、环境、时间范围和用途**的证据向量：

```text
问题有效
AND 数据与测量有效
AND 模型在适用域内充分
AND 选择过程无泄漏且可复算
AND 不确定性与失败边界可信
AND 外部环境中可泛化
AND claim 所需的机制或决策证据成立
AND 独立方可复现并签署
AND 现实行动另经人类治理
```

下层成立不能自动推出上层成立。因此系统必须输出 claim ceiling，而不是只输出“成功”。

## 当前全链能力矩阵

| 环节 | 当前能力 | 当前状态 | 主要边界 |
|---|---|---|---|
| 真实问题发现 | S0 合同、目标与边界可冻结 | PARTIAL | 仍缺开放世界问题发现、利益相关者冲突检查和现实需求验证 |
| 多向模型探索 | 四类隔离分支、盲探索、形式化审计、一次修复 | IMPLEMENTED | 分支类型和调用预算固定，不等于覆盖所有科学范式 |
| 分支知识迁移 | 内容寻址知识单元、broker 披露、定向翻译和经验隔离 | IMPLEMENTED | 同一基础模型的多进程不是独立科学复现 |
| 数据获取与测量 | 类型化 S2、World Bank 公共来源认证、变换与 provenance 绑定 | PARTIAL | 通用数据库、传感器、实验设计、缺失/偏差审计仍缺 |
| 模型可执行化 | 注册 IR、候选族、bundle 和 exact selected-model identity | IMPLEMENTED_NARROW | 仅两个正值标量序列 capability packs |
| 求解与模型选择 | ODE 与 adaptive candidate graph、确定性 replay | IMPLEMENTED_NARROW | 向量 ODE、PDE、网络、控制、优化、因果等仍是 capability gap |
| 科学检验 | L0--L4、rolling origin、基线、残差和区间覆盖 | IMPLEMENTED_NARROW | 只对已注册 adapter 的明确计算成立 |
| 多轮失败恢复 | 图撤销、attempt lineage、预算、重复失败停止、ODE→adaptive branch | IMPLEMENTED_NARROW | 不是开放式自动发明新骨架；未注册方向转 HUMAN |
| 报告与论文 | 结果/UQ 绑定、机器索引、S5 dossier、S6 PDF | IMPLEMENTED | 文档一致不等于科学真实性 |
| 当前模型外推 | 类型化未来坐标、真实 selected-model runtime、数值重算 replay | IMPLEMENTED_NARROW | 只支持当前两个 pack 的正值预测 |
| 资格事务协调 | 锁内 reopen→verify→project→intent→one action→receipt；预测与预约可崩溃恢复 | IMPLEMENTED_CANDIDATE | 已阻断底层预测/预约绕过并区分 V6.3 protocol flag；外部 ingress、failure receipt 与服务身份仍未完成 |
| 外部未见评价 | custody→registry→reservation→aggregate evaluation→promotion→replay | PROTOCOL_VALIDATED | 当前本地测试不是实际外部 campaign |
| 外部性证明 | root manifest、host attestation、revocation、exact replay 的 V6.4 本地机制 | PROTOCOL_PASS / NOT_RUN | 本地 provider 永不授予资格；真实 OS/KMS/HSM pinned trust root、远程节点和运维证据尚未产生 |
| 决策价值 | 冻结 decision contract 与回顾性证据 | PARTIAL | 没有 prospective intervention/regret trial |
| 机制/因果 | claim 维度和 fail-closed 状态已定义 | NOT_RUN | 没有结构可辨识、干预或独立机制 adapter |
| 部署与行动 | 永久区分 qualification 与 action | GOVERNED | 没有在线监控、漂移、回滚、事故响应和人类审批闭环 |

## 已关闭的关键成功判定漏洞

### 1. “有一个 external hash”不再等于预测对象已冻结

V6.3 使用不含私有目标值的 `ExternalForecastInputV63` 冻结 target IDs、顺序和预测时间。Custody、预测向量、注册、评价和最终 replay 必须绑定同一语义 hash。

### 2. “Harness 签过”不再等于数值确实来自当前模型

`external_prediction_runtime.py` 从当前 V6.2 public snapshot、sealed scientific bundle 和 executable receipt 重新拟合已选模型。工作区重开后会重新计算并逐值比较预测向量，同时验证固定 runtime trace、provider、adapter 和 role receipt。即使一个错误向量获得了本地 authority 签名，数值 replay 仍会拒绝。

### 3. 一个任务不能通过更换 qualification ID 获得第二次私测机会

同一 task、同一 artifact kind 的资格工件 create-once；预测输入、注册、评价 consumption 和 terminal receipt 均不可改写。Evaluation reservation 在远程派发前写入，失败或低分同样消耗唯一机会。

### 4. 预测时间顺序进入机械检查

强制：

```text
contract frozen
<= forecast input frozen
<= custody attested
<= generator receipt issued
<= registry registration
<= prediction seal
<= evaluation reservation
<= aggregate evaluation
<= promotion
```

不能在看到私有评价后补写预测或注册。

## 仍然阻断“完整真实建模成功”的缺口

### P0：一次真实预测资格必须关闭

1. 当前 coordinator candidate 已关闭预测与预约的 single-writer、CAS、intent/receipt、数值 replay 和两类崩溃恢复；仍须增加 custody/evaluation/promotion 的类型化 ingress、失败收据、人工 reconciliation 和受保护服务身份；
2. 外部 root、角色证书、host/runtime attestation、revocation epoch 必须来自任务调用者不可修改的部署能力；
3. 必须运行一个预先冻结、非 fixture、严格未见的真实任务；
4. custodian、registry、evaluator、promotion 必须是实际独立控制域，而不是四个本地 signer；
5. 当前真实工作区集成测试中被 monkeypatch 的 closure 与 S4/S6 必须换成完整真实链；
6. 远程 evaluator 必须使用 reservation hash 作为幂等键，并验证超时、不确定结果、崩溃恢复和取消语义；
7. 资格结果必须由纯读 replay 从 exact authority artifacts 重算，缺工件不得自动修复。

上述任何一项缺失，终态应保持 `NOT_RUN`，不能写成 external qualification。

### P1：从窄域预测器走向前沿建模 Agent

1. 增加 vector/nonautonomous ODE、PDE、network、agent-based、optimization/control、causal 和 hybrid mechanistic-data adapters；
2. 把模型骨架生成与注册规则结合起来，让开放式候选经过 compiler、verifier 和私测资格，而不是把 LLM 文本直接当模型；
3. 增加结构可辨识、观测可辨识、实验设计、主动采证和反事实判别组件；
4. 将模型不确定性、参数不确定性、结构不确定性、分布漂移和 open-set abstention 分开检验；
5. 支持多个独立环境、多个时间窗口和独立实现复现，而不是单次 heldout 阈值；
6. 为 prescriptive claim 建立 prospective decision trial、utility/regret、干预安全与伦理审批；
7. 把共享知识库从同任务 broker 扩展为经过 admission、冲突处理、时效和负迁移审计的长期经验库。

### P2：产品化和长期自治

1. 常驻 coordinator service、无密钥模型工具和只读 Studio 状态；
2. mTLS endpoint identity、KMS/HSM、证书轮换、撤销和操作审计；
3. 成本/算力预算、优先级、取消、暂停和人工接管；
4. 在线监控、漂移检测、shadow/canary、回滚和事故响应；
5. 多租户隔离、备份恢复、迁移和长期证据保留。

## 当前可接受的对外表述

> FMA 已实现一个 graph-native、证据分层、可恢复的正值标量时间序列建模纵切。它能并行探索候选、执行两个已注册预测 adapter、完成本地 L0--L4 与 rolling confirmation，并生成可重算的外部预测资格协议工件。当前证据证明本地协议与数值 replay 的实现，不证明真实外部资格已经发生，也不证明任意前沿建模问题可被自主解决。

V6.3 coordinator 中的 `EXTERNALLY_QUALIFIED` 只表示 V6.3 签名协议终态；其
`v63_protocol_qualification_granted` 可以为真，但
`scientific_qualification_granted` 永远为假。当前 V6.4 即使完整通过本地
manifest、revocation、host attestation 和 replay，也只输出
`anchor_protocol_status=PASS`、`status=NOT_RUN`。没有仓库外不可由任务构造的
部署锚时，前端和报告不得把这些状态显示为真实科学资格。

## 下一次资格实验的进入条件

只有以下冻结包全部就绪，才启动一个全新未见任务：

```text
真实 V6.2 current closure
+ exact selected-model runtime
+ externally pinned root-signed authority manifest
+ monotonic current revocation snapshot
+ 4-role host/runtime attestations
+ public forecast coordinates
+ private targets kept outside coordinator
+ one-shot metric/baseline/threshold
+ reservation-hash remote idempotency
+ immutable stop and claim rules
```

在外部方尚未提供签名工件时，可以准备 outbox 和验证工具，但实验状态必须是 `NOT_RUN`。

## 2026-07-28 工程验证快照

当前仓库共收集 92 个测试文件、557 个用例。单体 `python -m pytest -q`
在工具 1000 秒上限处被终止，因此不把该次运行计为通过或失败；随后按代际、
慢文件和 Studio nodeid 分片，全部取得明确 pytest 终态：

| 分片 | 结果 |
|---|---:|
| legacy + V2（含 Studio 22 个 nodeid） | 160 passed |
| V3 + V4 | 136 passed |
| V5 | 155 passed |
| V6.0--V6.4 | 106 passed |
| **合计** | **557 passed, 0 failed, 0 skipped** |

V6 源码和测试同时通过 Ruff、`compileall` 与 import smoke。长分片产生的
timeout 只作为执行诊断，之后均以更小分片复跑；确认父进程已退出的孤儿
pytest 被精确终止，没有删除或重写工作区证据。

该验证证明当前实现的工程回归状态，不证明真实外部 campaign 已运行，也不
改变 V6.4 的 `NOT_RUN`、机制资格、决策资格或现实行动边界。
