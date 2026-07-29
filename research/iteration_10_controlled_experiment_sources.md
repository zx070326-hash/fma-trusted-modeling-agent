# 大迭代 10：受控实验与目标导向 acquisition 证据冻结

冻结日期：2026-07-22（Asia/Shanghai）
用途：只支持 V3.1 合成 WorldPack 的设计选择；不构成现实系统安全、最优性或普适有效性证明。

## 检索契约

- 目标：寻找能约束“已知执行器下如何选择下一次输入实验”的原始研究。
- 范围：主动系统辨识、最优实验设计、目标导向实验设计、安全学习。
- 来源优先级：论文正式页面或 arXiv 原始条目；不使用二手博客支撑技术主张。
- arXiv 可复现查询（2026-07-22 访问）：
  `https://export.arxiv.org/api/query?id_list=2605.26093%2C2407.16212%2C2408.09582&start=0&max_results=3`
- 查询返回：`2407.16212v2`（更新 2026-04-29）、`2605.26093v1`、`2408.09582v1`。

## 采用的原始来源与可迁移约束

1. Ott, Kochenderfer, Boyd, *Informative Input Design for Dynamic Mode Decomposition*, L4DC 2025, PMLR 283:336–349.
   https://proceedings.mlr.press/v283/ott25a.html
   - 原始主张：可在状态与控制输入约束下，以近似凸优化降低参数估计协方差；其方法在论文实验中优于不适应当前模型的 PRBS/多正弦输入。
   - 本轮迁移：输入动作必须显式携带状态/控制约束和信息分数；固定随机输入是必要基线。
   - 不迁移：本文的线性/DMD 假设和理论性质不能外推到本轮非线性多项式漂移。

2. Mania, Jordan, Recht, *Active Learning for Nonlinear System Identification with Guarantees*, JMLR 23(32), 2022.
   https://jmlr.org/papers/v23/20-807.html
   - 原始主张：对状态—动作已知特征嵌入、参数线性的非线性动力系统，以规划—跟踪—重估的循环主动覆盖特征方向。
   - 本轮迁移：已知 actuator map；每次观察后重新拟合并重新选择实验。
   - 不迁移：本轮不满足其全部规划、可跟踪性和有限样本假设，不宣称其参数率保证。

3. Wagenmaker, Jamieson, *Active Learning for Identification of Linear Dynamical Systems*, COLT 2020, PMLR 125.
   https://proceedings.mlr.press/v125/wagenmaker20a.html
   - 原始主张：在完全控制输入的线性动力系统中，自适应输入可优于噪声激励，并给出有限时间与渐近结果。
   - 本轮迁移：同预算比较“预冻结随机输入”和“根据现有模型选择输入”。
   - 不迁移：本轮是非线性合成方言，不继承线性系统的最优率。

4. Huan, Jagalur, Marzouk, *Optimal experimental design: Formulations and computations*, arXiv:2407.16212v2.
   https://arxiv.org/abs/2407.16212
   - 原始主张：实验设计准则编码实验目标；现代 OED 包含离散/连续设计和顺序策略，并面临非线性、高维与高成本计算困难。
   - 本轮迁移：acquisition receipt 必须同时记录目标、信息、成本、风险和可执行性，不能只留一个综合分数。

5. Chakraborty, Huan, Catanach, *A Likelihood-Free Approach to Goal-Oriented Bayesian Optimal Experimental Design*, arXiv:2408.09582.
   https://arxiv.org/abs/2408.09582
   - 原始主张：参数 EIG 高不保证下游 QoI 的 EIG 高，目标导向 OED 直接面向预测 QoI。
   - 本轮迁移：实验分数由冻结的 `decision_target` 加权；未知目标先澄清，禁止假装“全局信息最大”就是任务最优。

6. Go, Qian, Yoon, *Goal-driven Bayesian Optimal Experimental Design for Robust Decision-Making Under Model Uncertainty*, arXiv:2605.26093v1, 2026.
   https://arxiv.org/abs/2605.26093
   - 原始主张：论文提出把后验代理接到鲁棒决策层，直接优化决策目标，并在三个案例中比较目标导向设计。
   - 本轮迁移：认识动作的终端评价使用决策相关 probe loss，而非仅用参数恢复。
   - 风险：这是 2026 年预印本；本轮只把它当候选架构证据，不视为已独立复现的事实。

7. Ahmadi et al., *Safely Learning Dynamical Systems from Short Trajectories*, L4DC 2021, PMLR 144.
   https://proceedings.mlr.press/v144/ahmadi21a.html
   - 原始主张：其安全定义要求所有与当前信息一致的动力系统在动作下保持于安全区；强结果限于线性短轨迹，非线性结果限于一次作用。
   - 本轮迁移：若没有可接受动作必须明确弃权；风险必须在执行前进入 permission。
   - 关键边界：V3.1 的 ensemble/bootstrap envelope 不等于“一致模型集合”证明，只能叫 `empirical_prediction_risk`。

## V3.1 预冻结设计

### 第一性原理状态

每个 episode 的可审计认识状态为：

`K_t = (mission, episode_contract, observations, model_ensemble, budget, permissions)`

策略只能提议动作。Harness 对有限目录中的动作重算：

`utility(a) = goal_information(a) + model_discrimination(a) - action_cost(a) - empirical_risk_penalty(a)`

Harness 再执行：schema → known-actuator binding → peak/energy/switch/cost → empirical-risk gate → budget → synthetic Reality Interface。

### 单一主消融

- baseline：预冻结的随机无放回输入；从不重构问题。
- candidate：目标导向信息/模型分歧 acquisition；若目标语义缺失，先花一步取得权威语义。
- 两臂共同：同一 pilot、同一模型库、同一估计器、同一观测噪声、同一两步总预算。
- 所有可执行输入：相同峰值、总能量、段数和切换数；只改变输入时序。
- actuator map `B` 由 WorldPack 冻结并公开；V3.1 禁止估计未知 `B`。

### 认识路由

- `problem_layer`：目标语义缺失，且有绑定来源的澄清动作。
- `data_layer`：确定性校准/数据质量门失败；禁止执行输入。
- `model_layer`：数据门通过但冻结模型库的持出残差持续越界。
- `none`：上述门均不触发。

### 明确不宣称

- 不宣称现实安全、形式化可达性、结构可识别性或未知 actuator 学习。
- 不把 synthetic private WorldPack 通过外推为开放世界自主建模。
- 不允许策略、模型或综合 utility 自己批准动作、证据或 qualification。
