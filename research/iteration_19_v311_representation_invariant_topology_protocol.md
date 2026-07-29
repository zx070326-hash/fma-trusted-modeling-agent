# 大迭代 19：V3.11 表示不变拓扑发现确认协议

状态：`FROZEN_BEFORE_CONFIRMATION_PRIVATE_WORLDPACK`

冻结日期：2026-07-22

## 要回答的单一问题

V3.10 的成功可能只是 fixture shortcut：公开变量名直接泄露 `position/velocity` 语义，机制也来自同一受控动力学包。V3.11 只检验一个更窄、可证伪的问题：

> 在状态名匿名、坐标可置换、数值单位可缩放、机制跨域且含候选库外系统时，只读公开轨迹的拓扑假设器，能否稳定识别受支持拓扑，并对库外机制正确弃权？

它不检验部分观测、不规则采样、隐变量、随机动力学、PDE、因果识别或真实世界有效性。这些因素本轮不同时加入，以避免无法归因的多因素变化。

## 方法证据与检索范围

检索于 2026-07-22 使用 OpenAlex API，每个查询取一页并做目标论文精确核对；这是针对性检索，不是系统综述：

| 查询 | OpenAlex count | 借用的设计原则 |
|---|---:|---|
| `data driven discovery coordinates governing equations dynamical systems` | 8782 | 坐标选择和控制方程发现应作为耦合假设检验 |
| `constrained sparse identification nonlinear dynamics conservation laws` | 3518 | 将物理等式作为回归约束，与无约束候选比较 |
| `dimensional analysis symmetry symbolic regression physical laws` | 3334 | 用一般物理约束缩小搜索，但保留独立预测检验 |

冻结的一手来源：

- Champion et al. (2019), *Data-driven discovery of coordinates and governing equations*, DOI `10.1073/pnas.1906995116`。
- Loiseau & Brunton (2018), *Constrained sparse Galerkin regression*, DOI `10.1017/jfm.2017.823`。
- Reinbold et al. (2021), *Robust learning from noisy, incomplete, high-dimensional experimental data via physically constrained symbolic regression*, DOI `10.1038/s41467-021-23479-0`。

这些论文只提供方法动机；其理论、实验结论和适用范围不迁移为 FMA 的证据。方法证据哈希：`3eb2eddea8cb2de46713b32e27ac928694cb88d2cdafd1d42635b0ed6dceb6d5`。

## 公开/私有隔离

Generator 只能接收：

- `PublicTopologyProtocolV311`；
- `PublicTopologyWorldPackV311`；
- `TopologyDiscoveryPolicyV311`。

每个公开 case 只有匿名状态名 `z0/z1/z2`、三条 61 点轨迹、时间和公开质量标志。它看不到机制、representation、成对 ID、物理参数、坐标置换、单位缩放、私有 OOD 初值/真值和 target loss。

Harness 私有保存这些字段，生成和评价使用不同 typed objects。公开执行函数签名不接受 private world pack。task router、模型 qualification 和现实执行权限保持 `false`。

## 冻结变化与对照

相对 V3.10，本轮只引入表示与机制变化：

1. 所有语义状态名匿名化；
2. 同一隐藏物理 case 生成 reference 与 scaled/permuted 成对表示；
3. 新增 thermal relaxation、Van der Pol、Lotka–Volterra、SIR 四种受支持机制；
4. 新增 candidate library 明确不含 `sin(theta)` 的大振幅 pendulum open-set sentinel；
5. topology hypothesis 与最终 prediction switch 分开记录：公开 switch guard 可为预测保留 generic baseline，但不得抹掉结构假设；
6. 不加入部分观测或不规则采样。

旧基线为匿名坐标上的 `generic_cubic`。候选目录固定为：

- `generic_cubic`；
- `scalar_affine_rate`；
- `second_order_kinematic`；
- `interacting_population`；
- `conserved_compartment`；
- `uncoupled_linear_decoy`。

预期拓扑分别是 rate、kinematic、population、compartment；pendulum 的预期是 `None`，即正确的 open-set 弃权。候选库内不存在可精确表达摆方程的 sine skeleton。

## 公开拟合与选择规则

- 每 case 三条公开轨迹，61 点，`dt=0.03`；
- window integral fitting，窗口 6 个 interval，ridge `1e-4`；
- blocked tail 从 index 38 开始；
- rank ratio 至少 `0.9`，condition number 至多 `1e9`；
- public CV loss 最大 `0.35`，否则 topology hypothesis 弃权；
- 候选按双验证 minimax、稀疏性/简约性选择；
- prediction switch 仍需 public leave-one-trajectory-out guard；
- 未解析私有损失固定为 `10.0`。

## 开发/确认分离

开发 seeds 固定为 `26003, 26069, 26141, 26203, 26267, 26321`。共 60 cases，其中第一 seed 的 10 cases 只有公开质量失败，不进入性能分母；性能 50、open-set 10。

开发前第一次运行使用了 `14:00Z` 的逻辑时间，而当时本机真实 UTC 约为 `04:37Z`。该运行虽内容哈希和科学门完整，但审计时间在未来，已被当前 verifier 拒绝且不作为确认 lineage。它没有生成任何确认私有包。Harness 随即新增两条代码门：依赖冻结时间必须单调、审计时间不得比当前墙钟超前五分钟。

时间恢复版开发报告：

- status：`representation_topology_development_diagnostic_v311`；
- run：`v311_representation_topology_development_time_recovered`；
- public protocol hash：`de57f040e16baa2615681f7b4a3224a34b88a3a4041e533cc1fd1174315fafa6`；
- report hash：`28e43334cea22f5b8bd731592a950e8f6301dbf70e659636a3b90b761883d12b`；
- 13/13 冻结门通过，但 `ready_for_next_confirmation=false`；
- 该哈希必须写入确认 private spec，确认 verifier 会先独立复验开发包。

确认 seeds 在代码中预先固定为 `27011, 27077, 27143, 27211, 27277, 27329, 27397, 27457`，此前不参与阈值和结构选择。共 80 cases：10 个公开质量弃权、70 个性能 cases、其中 14 个 open-set、56 个受支持机制 cases。

## 正式确认门

所有门必须同时通过：

1. 公开质量分区恰为 10，候选矩阵完整；
2. 公开执行没有使用 private 字段或语义状态标签；
3. 受支持机制 topology coverage 至少 `0.90`；
4. 包含 open-set 在内的 topology accuracy 至少 `0.80`；
5. pendulum open-set 弃权率至少 `0.90`；
6. 成对 topology consistency 至少 `0.90`；
7. 成对表示的最大 private target-loss 差不超过 `0.03`；
8. 2000 次配对 bootstrap 的 mean-improvement 95% CI 下界大于 0；
9. 每个机制 candidate loss 不超过 baseline `+0.02`；
10. transformed 表示均值不超过 reference `+0.02`；
11. 材料性负迁移定义为 candidate loss 比 baseline 高 `>0.02`，其一侧 95% beta 上界不超过 `0.10`；
12. 不产生 task router、qualification 或现实执行权限。

确认包通过则终态仅为 `representation_topology_confirmed_v311`；失败则必须保留为 `representation_topology_refuted_v311`。无论结果如何，都不得用确认数据回调本版本；后续修改必须使用新版本和全新 seeds。

## 解释边界

整体均值会受到 baseline 对 SIR/部分 Van der Pol case 弃权、以及 open-set 双方都记未解析损失 10 的影响，因此不是“真实建模能力”的充分统计量。解释优先级依次是：公开/私有隔离、拓扑准确率、open-set 弃权、成对表示一致性、逐机制损失、负迁移上界，最后才是宏平均改善。

即使全通过，也只证明冻结合成 WorldPack 中的匿名坐标、置换、单位缩放和五机制表现；不证明自主发现任意方程、外部有效性或独立解决真实前沿建模问题。
