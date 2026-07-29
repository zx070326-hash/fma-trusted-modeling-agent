# 大迭代 8：主动实验设计证据契约

冻结日期：2026-07-22

## 要解决的瓶颈

大迭代 7 已确认：改变点导数/窗口积分估计器不能稳定改善域外轨迹与反事实表现；同一批被动轨迹可以提高结构恢复，却仍不足以约束长期行为。因此 V2.6 只检验一个新变量：**实验动作如何选择**。

可证伪问题：在候选初值、安全边界、实验次数、每次轨迹长度、观测噪声、候选库、拟合器、评测预算完全相同的条件下，基于模型集分歧的顺序初值干预，是否优于预冻结的随机安全初值选择？

## 原始来源与允许主张

1. Fasel 等，*Ensemble-SINDy*，Proceedings of the Royal Society A 478 (2022)，DOI `10.1098/rspa.2021.0904`，arXiv `2111.10992`。
   - 允许：bootstrap 模型集可给出候选项纳入概率和经验不确定性；作者展示了这些统计量可用于主动学习。
   - 不允许：把模型集方差当成校准后的真实世界风险概率。

2. Larrañaga、Fasel、Brunton，*How Low Can You Go? Active Learning for Sparse Model Discovery in the Ultra-Low-Data Limit*，arXiv `2606.12182v1`（2026-06-10）。
   - 允许：对 ODE，可从冻结候选池顺序选择使 E-SINDy 预测分歧最大的初始条件，采集短轨迹后重拟合；候选去重/多样性过滤和最大预算是协议的一部分。
   - 允许：作者的 Hopf 示例说明仅沿吸引子采样会使经验库矩阵严重病态，而扩展初值覆盖可改善条件数。
   - 不允许：该单篇新预印本不能证明这种策略跨机制、跨噪声或真实实验必然优于随机/D-optimal 设计。

3. Wagenmaker 与 Jamieson，*Active Learning for Identification of Linear Dynamical Systems*，COLT/PMLR 125 (2020)。
   - 允许：在论文的线性、完全可控输入及其假设下，自适应输入可加速参数估计，并存在有限时间分析。
   - 不允许：把线性系统的最优性定理外推到当前非线性稀疏方程发现。

4. Buisson-Fenet、Solowjow、Trimpe，*Actively Learning Gaussian Process Dynamics*，L4DC/PMLR 120 (2020)。
   - 允许：主动学习策略应显式服从系统动力学施加的采样约束。
   - 不允许：把 GP 信息论准则的性质自动转移给本实现的 bootstrap 稀疏回归。

## V2.6 冻结边界

- 动作不是任意连续控制，而是从 Harness 生成并验证的有限安全初值干预目录中选一个 reset；这是首个受控实验方言。
- 生成器只看到公开 pilot 轨迹、已观测轨迹和动作目录；看不到隐藏参数、干净轨迹、域外 probe 或真值支持。
- Harness 拥有预算、动作合法性、无重复、观测执行、停止条件、时间戳、哈希与重放。
- 基线与候选的唯一差异是动作选择规则：`random_safe_catalog` 对比 `ensemble_disagreement_catalog`。
- 主要终点是隐藏 probe 上的参数/支持/行为联合损失；经验库条件数只是诊断，不是结构可辨识性证明。
- exploratory 只允许修改一个组件后再测；confirmation 使用不相交 seed，并且协议在生成 private pack 前冻结。
- 即使通过，也只产生 `synthetic_safe_initial_condition_design_v26` 资格；不产生真实设备控制、通用自治建模或参数唯一性主张。

## 检索可复现性

- 数据库：arXiv Atom API，targeted lookup。
- endpoint：`https://export.arxiv.org/api/query`
- 参数：`id_list=2111.10992,2606.12182`，`max_results=2`
- 返回：2/2 个目标记录；访问日期 2026-07-22。
- 补充原始页面：
  - https://arxiv.org/abs/2111.10992
  - https://arxiv.org/abs/2606.12182
  - https://proceedings.mlr.press/v125/wagenmaker20a.html
  - https://proceedings.mlr.press/v120/buisson-fenet20a.html
