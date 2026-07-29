# 大迭代 1：方法证据与架构吸收记录

访问日期：2026-07-22。以下只记录能约束实现的来源；厂商能力描述视为系统设计证据，不视为开放世界科学能力证明。

## 采用

1. **滚动时间原点验证**
   - 来源：[Forecasting: Principles and Practice — Time series cross-validation](https://otexts.com/fpp3/tscv.html)
   - 证据：每个测试点的训练集只能包含它之前的观测；多个测试点形成 rolling forecasting origin，误差在这些持出点上汇总。
   - 落地：冻结 expanding-window 切分；evaluator 在每个原点重新拟合；禁止随机打乱和未来数据泄漏。

2. **预测必须连同不确定性表达**
   - 来源：[NIST/SEMATECH e-Handbook — Prediction](https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd132.htm)
   - 证据：单独的点预测不足以和目标或其他预测比较，预测应包含不确定性评估。
   - 落地：每个候选必须输出区间、持出覆盖率与区间宽度；没有足够历史残差时不能通过。

3. **验证、确认与 UQ 分离；可信度绑定用途**
   - 来源：[ASME VVUQ 概览](https://www.asme.org/codes-standards/publications-information/verification-validation-uncertainty)、[ASME VVUQ 1](https://www.asme.org/codes-standards/find-codes-standards/verification-validation-and-uncertainty-quantification-terminology-in-computational-modeling-and-simulation)、[NASA-STD-7009B](https://standards.nasa.gov/standard/nasa/nasa-std-7009)
   - 证据：verification 检查数学描述的实现，validation 检查对现实应用的代表性，UQ 检查参数/输入变化对结果的影响；NASA 要求由项目定义并批准 acceptance criteria。
   - 落地：本轮经验报告不覆盖 legacy 计算 verification；经验 validation 只能产生回顾性 shadow 证据；用途字段和现实行动授权分离。

4. **候选生成、反思/排名、演化是可借鉴算子，不是新权威主体**
   - 来源：[Google Research AI co-scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)
   - 证据：系统使用 generation、reflection、ranking、evolution 等循环改善假设。
   - 落地：保留“多候选—独立评价—有证据的修订”算子，但当前顺序闭环不引入固定人格式多 agent；评估权仍在代码拥有的门。

5. **长程探索应是可中止的树搜索，但论文产出不等于真实建模可信**
   - 来源：[The AI Scientist-v2](https://arxiv.org/abs/2504.08066)
   - 证据：使用 progressive agentic tree search 和实验管理器完成机器学习实验/论文循环。
   - 落地：后续把候选谱系设计成分支和 checkpoint；本轮尚无搜索需求，不引入树调度器，也不把 workshop 论文接收外推为现实世界有效性。

6. **多 agent 只在任务可并行且评测证明收益时采用**
   - 来源：[Google Research — Towards a science of scaling agent systems](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
   - 证据：其受控实验报告，多 agent 对可并行任务更有利、对顺序任务可能退化。
   - 落地：本轮保持一个控制循环，生成器和 evaluator 是权限分离的任务/模块；以后只有消融显示收益才增加 worker。

7. **分布预测要与基线在同一持出上比较，并报告尺度与区间**
   - 来源：[Forecasting: Principles and Practice — Distributional forecast accuracy](https://otexts.com/fpp3/distaccuracy.html)
   - 证据：分布预测需要量化区间/分位数质量；skill score 必须相对同一训练边界的基准，测试集过小时分母不稳定。
   - 落地：大迭代 2 使用同一 rolling-origin 目标上的配对绝对误差差值，并用三 seed circular moving-block bootstrap 报告效应区间；不把小样本 p 值作为晋级门。

8. **官方 API 只是来源，不代表数据不会修订**
   - 来源：[BLS Public Data API](https://www.bls.gov/developers/home.htm)、[BLS API signatures](https://www.bls.gov/developers/api_signature_v2.htm)、[USGS Daily Values Service](https://waterservices.usgs.gov/docs/dv-service/daily-values-service-details/)
   - 证据：BLS 提供公开历史时间序列 API；USGS Daily Values 支持按 site、parameter、statistic、date range 请求，且官方文档提醒数据状态与服务错误处理。
   - 落地：端点和主机 allowlist、精确请求身份、HTTP/UTF-8/JSON/series identity/连续频率检查、原始响应 hash 与 revision-prone warning；重放不重新联网，而从冻结原始响应复算。

## 暂不采用

- 不采用普通 split conformal 的无条件覆盖承诺。时间依赖破坏 exchangeability；本轮区间仅称“历史滚动残差区间”，并以未来持出覆盖率经验检查。
- 不采用模型自评作为晋级依据。
- 不采用“最终论文/报告看起来完整”作为建模成功指标；本轮首先测错误晋级、弃权、持出覆盖、regret 与可重放性。
- 不采用固定多 agent 组织图作为默认架构；它不是第一性原理需求，只有被可复现实验支持后才增加。
